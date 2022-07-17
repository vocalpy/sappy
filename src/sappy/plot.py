"""Plotting functions for ``sappy``."""
import numpy as np
import seaborn as sns
import matplotlib.patches as p
import matplotlib.pyplot as plt

from . import songfeatures


def spectral_derivs(spec_der, contrast=0.1, ax=None, freq_range=None,
                    ov_params=None):
    """Plot spectral derivative in greyscale.

    spec_der - The spectral derivatives of the song (computed with
               ``sappy.songfeatures.spectral_derivs``) or the song itself
    contrast - The contrast of the plot
    ax - The matplotlib axis where the plot must be drawn, if None, a new axis
         is created
    freq_range - The amount of frequency to plot, usefull only if `spec_der` is
                 a song. Given to `spectral_derivs`
    ov_params - The Parameters to override, passed to `spectral_derivs`
    """
    if spec_der.ndim == 1:
        spec_der = songfeatures.spectral_derivs(spec_der, freq_range, ov_params)
    ax = sns.heatmap(spec_der.T, yticklabels=100, xticklabels=100,
                     vmin=-contrast, vmax=contrast, ax=ax, cmap='Greys',
                     cbar=False)
    ax.invert_yaxis()
    return ax


def features_over_spec(data, ax, freq_range=256, **plot_params):
    """Plot an acoustic feature over a spectral derivatives plot.

    The data are first normalized, then rescaled to fit the ylim of the axis.
    """
    # Normalize the data so that they fit in the graph
    ndata = data / (np.nanmax(data) - np.nanmin(data))
    # We plot -ndata because the yaxis is inverted (see ``sappy.plot.spectral_derivs``)
    # We take for abscisse axis 95% of freq_range
    # We rescale the data so that they take 75% of the graph
    ax.plot(95/100 * freq_range - 75/100 * freq_range
            * (ndata - np.nanmin(ndata)), **plot_params)
    return ax


def similarity(sim, song, refsong):
    """Plot the result of ``sappy.similarity``."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 13),
                           gridspec_kw={'width_ratios': [1, 4],
                                        'height_ratios': [1, 4]})
    ax[0, 0].axis('off')
    sds = songfeatures.spectral_derivs(song)
    sdr = songfeatures.spectral_derivs(refsong)
    ax[0, 1] = spectral_derivs(sds, 0.05, ax[0, 1])
    ax[0, 1].set_title('Song')
    ax[1, 0] = spectral_derivs(np.flip(sdr.T, 1), 0.05, ax[1, 0])
    ax[1, 0].set_title('Reference Song')
    ax[1, 1] = sns.heatmap(sim['glob_matrix'], ax=ax[1, 1], cbar=False,
                           vmin=0, vmax=1)
    for section in sim['sections']:
        xy = (section['beg'][0],
              sim['glob_matrix'].shape[1] - section['end'][1])
        width = section['end'][0] - section['beg'][0]
        height = section['end'][1] - section['beg'][1]
        ax[1, 1].add_patch(p.Rectangle(xy, width, height, fill=False,
                                       edgecolor='y', linewidth=3))
    return fig
