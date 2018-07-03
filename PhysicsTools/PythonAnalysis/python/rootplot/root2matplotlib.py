"""
Utilities for plotting ROOT histograms in matplotlib.
"""

__license__ = '''\
Copyright (c) 2009-2010 Jeff Klukas <klukas@wisc.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

################ Import python libraries

import math
import ROOT
import re
import copy
import array
from rootplot import utilities
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

################ Define constants

_all_whitespace_string = re.compile(r'\s*$')


################ Define classes

class Hist2D(utilities.Hist2D):
    """A container to hold the parameters from a 2D ROOT histogram."""
    def __init__(self, *args, **kwargs):
        self.replacements = None
        if 'replacements' in kwargs:
            self.replacements = kwargs.pop('replacements')
        utilities.Hist2D.__init__(self, *args, **kwargs)
    def contour(self, **kwargs):
        """Draw a contour plot."""
        cs = plt.contour(self.x, self.y, self.content, **kwargs)
        plt.clabel(cs, inline=1, fontsize=10)
        if self.binlabelsx is not None:
            plt.xticks(np.arange(self.nbinsx), self.binlabelsx)
        if self.binlabelsy is not None:
            plt.yticks(np.arange(self.nbinsy), self.binlabelsy)
        return cs
    def col(self, **kwargs):
        """Draw a colored box plot using :func:`matplotlib.pyplot.imshow`."""
        if 'cmap' in kwargs:
            kwargs['cmap'] = plt.get_cmap(kwargs['cmap'])
        plot = plt.imshow(self.content, interpolation='nearest',
                          extent=[self.xedges[0], self.xedges[-1],
                                  self.yedges[0], self.yedges[-1]],
                          aspect='auto', origin='lower', **kwargs)
        return plot
    def colz(self, **kwargs):
        """
        Draw a colored box plot with a colorbar using
        :func:`matplotlib.pyplot.imshow`.
        """
        plot = self.col(**kwargs)
        plt.colorbar(plot)
        return plot
    def box(self, maxsize=40, **kwargs):
        """
        Draw a box plot with size indicating content using
        :func:`matplotlib.pyplot.scatter`.
        
        The data will be normalized, with the largest box using a marker of
        size maxsize (in points).
        """
        x = np.hstack([self.x for i in range(self.nbinsy)])
        y = np.hstack([[yval for i in range(self.nbinsx)] for yval in self.y])
        maxvalue = np.max(self.content)
        if maxvalue == 0:
            maxvalue = 1
        sizes = np.array(self.content).flatten() / maxvalue * maxsize
        plot = plt.scatter(x, y, sizes, marker='s', **kwargs)
        return plot
    def TH2F(self, name=""):
        """Return a ROOT.TH2F object with contents of this Hist2D."""
        th2f = ROOT.TH2F(name, "",
                         self.nbinsx, array.array('f', self.xedges),
                         self.nbinsy, array.array('f', self.yedges))
        th2f.SetTitle("%s;%s;%s" % (self.title, self.xlabel, self.ylabel))
        for ix in range(self.nbinsx):
            for iy in range(self.nbinsy):
                th2f.SetBinContent(ix + 1, iy + 1, self.content[iy][ix])
        return th2f

class Hist(utilities.Hist):
    """A container to hold the parameters from a ROOT histogram."""
    def __init__(self, *args, **kwargs):
        self.replacements = None
        if 'replacements' in kwargs:
            self.replacements = kwargs.pop('replacements')
        utilities.Hist.__init__(self, *args, **kwargs)
    def _prepare_xaxis(self, rotation=0, alignment='center'):
        """Apply bounds and text labels on x axis."""
        if self.binlabels is not None:
            binwidth = (self.xedges[-1] - self.xedges[0]) / self.nbins
            plt.xticks(self.x, self.binlabels,
                       rotation=rotation, ha=alignment)
        plt.xlim(self.xedges[0], self.xedges[-1])

    def _prepare_yaxis(self, rotation=0, alignment='center'):
        """Apply bounds and text labels on y axis."""
        if self.binlabels is not None:
            binwidth = (self.xedges[-1] - self.xedges[0]) / self.nbins
            plt.yticks(self.x, self.binlabels,
                       rotation=rotation, va=alignment)
        plt.ylim(self.xedges[0], self.xedges[-1])

    def show_titles(self, **kwargs):
        """Print the title and axis labels to the current figure."""
        replacements = kwargs.get('replacements', None) or self.replacements
        plt.title(replace(self.title, replacements))
        plt.xlabel(replace(self.xlabel, replacements))
        plt.ylabel(replace(self.ylabel, replacements))
    def hist(self, label_rotation=0, label_alignment='center', **kwargs):
        """
        Generate a matplotlib hist figure.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.hist`.
        """
        kwargs.pop('fmt', None)
        replacements = kwargs.get('replacements', None) or self.replacements
        weights = self.y
        # Kludge to avoid mpl bug when plotting all zeros
        if self.y == [0] * self.nbins:
            weights = [1.e-10] * self.nbins
        plot = plt.hist(self.x, weights=weights, bins=self.xedges,
                        label=replace(self.label, replacements), **kwargs)
        self._prepare_xaxis(label_rotation, label_alignment)
        return plot
    def errorbar(self, xerr=False, yerr=False, label_rotation=0,
                 label_alignment='center', **kwargs):
        """
        Generate a matplotlib errorbar figure.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.errorbar`.
        """
        if xerr:
            kwargs['xerr'] = self.xerr
        if yerr:
            kwargs['yerr'] = self.yerr
        replacements = kwargs.get('replacements', None) or self.replacements
        errorbar = plt.errorbar(self.x, self.y,
                                label=replace(self.label, replacements),
                                **kwargs)
        self._prepare_xaxis(label_rotation, label_alignment)
        return errorbar
    def errorbarh(self, xerr=False, yerr=False, label_rotation=0,
                  label_alignment='center', **kwargs):
        """
        Generate a horizontal matplotlib errorbar figure.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.errorbar`.
        """
        if xerr: kwargs['xerr'] = self.yerr
        if yerr: kwargs['yerr'] = self.xerr
        replacements = kwargs.get('replacements', None) or self.replacements
        errorbar = plt.errorbar(self.y, self.x,
                                label=replace(self.label, replacements),
                                **kwargs)
        self._prepare_yaxis(label_rotation, label_alignment)
        return errorbar
    def bar(self, xerr=False, yerr=False, xoffset=0., width=0.8, 
            label_rotation=0, label_alignment='center', **kwargs):
        """
        Generate a matplotlib bar figure.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.bar`.
        """
        kwargs.pop('fmt', None)
        if xerr: kwargs['xerr'] = self.av_xerr()
        if yerr: kwargs['yerr'] = self.av_yerr()
        replacements = kwargs.get('replacements', None) or self.replacements
        ycontent = [self.xedges[i] + self.width[i] * xoffset
                    for i in range(len(self.xedges) - 1)]
        width = [x * width for x in self.width]
        bar = plt.bar(ycontent, self.y, width,
                      label=replace(self.label, replacements), **kwargs)
        self._prepare_xaxis(label_rotation, label_alignment)
        return bar
    def barh(self, xerr=False, yerr=False, yoffset=0., width=0.8,
             label_rotation=0, label_alignment='center', **kwargs):
        """
        Generate a horizontal matplotlib bar figure.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.bar`.
        """
        kwargs.pop('fmt', None)
        if xerr: kwargs['xerr'] = self.av_yerr()
        if yerr: kwargs['yerr'] = self.av_xerr()
        replacements = kwargs.get('replacements', None) or self.replacements
        xcontent = [self.xedges[i] + self.width[i] * yoffset
                    for i in range(len(self.xedges) - 1)]
        width = [x * width for x in self.width]
        barh = plt.barh(xcontent, self.y, width,
                        label=replace(self.label, replacements),
                       **kwargs)
        self._prepare_yaxis(label_rotation, label_alignment)
        return barh

class HistStack(utilities.HistStack):
    """
    A container to hold Hist objects for plotting together.

    When plotting, the title and the x and y labels of the last Hist added
    will be used unless specified otherwise in the constructor.
    """
    def __init__(self, *args, **kwargs):
        if 'replacements' in kwargs:
            self.replacements = kwargs.pop('replacements')
        utilities.HistStack.__init__(self, *args, **kwargs)
    def show_titles(self, **kwargs):
        self.hists[-1].show_titles()
    def hist(self, label_rotation=0, **kwargs):
        """
        Make a matplotlib hist plot.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.hist`, which allows a vast array of
        possibilities.  Particlularly, the *histtype* values such as
        ``'barstacked'`` and ``'stepfilled'`` give substantially different
        results.  You will probably want to include a transparency value
        (i.e. *alpha* = 0.5).
        """
        contents = np.dstack([hist.y for hist in self.hists])
        xedges = self.hists[0].xedges
        x = np.dstack([hist.x for hist in self.hists])[0]
        labels = [hist.label for hist in self.hists]
        try:
            clist = [item['color'] for item in self.kwargs]
            plt.gca().set_color_cycle(clist)
            ## kwargs['color'] = clist # For newer version of matplotlib
        except:
            pass
        plot = plt.hist(x, weights=contents, bins=xedges,
                        label=labels, **kwargs)
    def bar3d(self, **kwargs):
        #### Not yet ready for primetime
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        plots = []
        labels = []
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.xlabel = self.xlabel
            if self.ylabel is not None: hist.ylabel = self.ylabel
            labels.append(hist.label)
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            bar = ax.bar(hist.x, hist.y, zs=i, zdir='y', width=hist.width,
                         **all_kwargs)
            plots.append(bar)
        from matplotlib.ticker import FixedLocator
        locator = FixedLocator(range(len(labels)))
        ax.w_yaxis.set_major_locator(locator)
        ax.w_yaxis.set_ticklabels(labels)
        ax.set_ylim3d(-1, len(labels))
        return plots
    def barstack(self, **kwargs):
        """
        Make a matplotlib bar plot, with each Hist stacked upon the last.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.bar`.
        """
        bottom = None # if this is set to zeroes, it fails for log y
        plots = []
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.xlabel = self.xlabel
            if self.ylabel is not None: hist.ylabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            bar = hist.bar(bottom=bottom, **all_kwargs)
            plots.append(bar)
            if not bottom: bottom = [0. for i in range(self.hists[0].nbins)]
            bottom = [sum(pair) for pair in zip(bottom, hist.y)]
        return plots
    def histstack(self, **kwargs):
        """
        Make a matplotlib hist plot, with each Hist stacked upon the last.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.hist`.
        """
        bottom = None # if this is set to zeroes, it fails for log y
        plots = []
        cumhist = None
        for i, hist in enumerate(self.hists):
            if cumhist:
                cumhist = hist + cumhist
            else:
                cumhist = copy.copy(hist)
            if self.title  is not None: cumhist.title  = self.title
            if self.xlabel is not None: cumhist.xlabel = self.xlabel
            if self.ylabel is not None: cumhist.ylabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            zorder = 0 + float(len(self) - i)/len(self) # plot in reverse order
            plot = cumhist.hist(zorder=zorder, **all_kwargs)
            plots.append(plot)
        return plots
    def barcluster(self, width=0.8, **kwargs):
        """
        Make a clustered bar plot.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.bar`.
        """
        plots = []
        spacer = (1. - width) / 2
        width = width / len(self.hists)
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.xlabel = self.xlabel
            if self.ylabel is not None: hist.ylabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            bar = hist.bar(xoffset=width*i + spacer, width=width, **all_kwargs)
            plots.append(bar)
        return plots
    def barh(self, width=0.8, **kwargs):
        """
        Make a horizontal clustered matplotlib bar plot.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.bar`.
        """
        plots = []
        spacer = (1. - width) / 2
        width = width / len(self.hists)
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.ylabel = self.xlabel
            if self.ylabel is not None: hist.xlabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            bar = hist.barh(yoffset=width*i + spacer, width=width, **all_kwargs)
            plots.append(bar)
        return plots
    def bar(self, **kwargs):
        """
        Make a bar plot, with all Hists in the stack overlaid.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.bar`.  You will probably want to set a 
        transparency value (i.e. *alpha* = 0.5).
        """
        plots = []
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.xlabel = self.xlabel
            if self.ylabel is not None: hist.ylabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            bar = hist.bar(**all_kwargs)
            plots.append(bar)
        return plots
    def errorbar(self, offset=False, **kwargs):
        """
        Make a matplotlib errorbar plot, with all Hists in the stack overlaid.

        Passing 'offset=True' will slightly offset each dataset so overlapping
        errorbars are still visible.  Any additional keyword arguments will
        be passed to :func:`matplotlib.pyplot.errorbar`.
        """
        plots = []
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.xlabel = self.xlabel
            if self.ylabel is not None: hist.ylabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            transform = plt.gca().transData
            if offset:
                index_offset = (len(self.hists) - 1)/2.
                pixel_offset = 1./72 * (i - index_offset)
                transform = transforms.ScaledTranslation(
                    pixel_offset, 0, plt.gcf().dpi_scale_trans)
                transform = plt.gca().transData + transform
            errorbar = hist.errorbar(transform=transform, **all_kwargs)
            plots.append(errorbar)
        return plots
    def errorbarh(self, **kwargs):
        """
        Make a horizontal matplotlib errorbar plot, with all Hists in the
        stack overlaid.

        Any additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.errorbar`.
        """
        plots = []
        for i, hist in enumerate(self.hists):
            if self.title  is not None: hist.title  = self.title
            if self.xlabel is not None: hist.ylabel = self.xlabel
            if self.ylabel is not None: hist.xlabel = self.ylabel
            all_kwargs = copy.copy(kwargs)
            all_kwargs.update(self.kwargs[i])
            errorbar = hist.errorbarh(**all_kwargs)
            plots.append(errorbar)
        return plots

################ Define functions and classes for navigating within ROOT

class RootFile(utilities.RootFile):
    """A wrapper for TFiles, allowing easier access to methods."""
    def get(self, object_name, path=None):
        try:
            return utilities.RootFile.get(self, object_name, path,
                                          Hist, Hist2D)
        except ReferenceError as e:
            raise ReferenceError(e)

################ Define additional helping functions

def replace(string, replacements):
    """
    Modify a string based on a list of patterns and substitutions.

    replacements should be a list of two-entry tuples, the first entry giving
    a string to search for and the second entry giving the string with which
    to replace it.  If replacements includes a pattern entry containing
    'use_regexp', then all patterns will be treated as regular expressions
    using re.sub.
    """
    if not replacements:
        return string
    if 'use_regexp' in [x for x,y in replacements]:
        for pattern, repl in [x for x in replacements
                              if x[0] != 'use_regexp']:
            string = re.sub(pattern, repl, string)
    else:
        for pattern, repl in replacements:
            string = string.replace(pattern, repl)
    if re.match(_all_whitespace_string, string):
        return ""
    return string

