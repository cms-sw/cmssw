"""
Utilities for rootplot including histogram classes.
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
import os.path

################ Define classes

class Hist2D(object):
    """A container to hold the parameters from a 2D ROOT histogram."""
    def __init__(self, hist, label="__nolabel__", title=None,
                 xlabel=None, ylabel=None):
        try:
            if not hist.InheritsFrom("TH2"):
                raise TypeError("%s does not inherit from TH2" % hist)
        except:
            raise TypeError("%s is not a ROOT object" % hist)
        self.rootclass = hist.ClassName()
        self.name = hist.GetName()
        self.nbinsx = nx = hist.GetNbinsX()
        self.nbinsy = ny = hist.GetNbinsY()
        self.binlabelsx = process_bin_labels([hist.GetXaxis().GetBinLabel(i)
                                               for i in range(1, nx + 1)])
        if self.binlabelsx:
            self.nbinsx = nx = self.binlabelsx.index('')
            self.binlabelsx = self.binlabelsx[:ny]
        self.binlabelsy = process_bin_labels([hist.GetYaxis().GetBinLabel(i)
                                               for i in range(1, ny + 1)])
        if self.binlabelsy:
            self.nbinsy = ny = self.binlabelsy.index('')
            self.binlabelsy = self.binlabelsy[:ny]
        self.entries = hist.GetEntries()
        self.content = [[hist.GetBinContent(i, j) for i in range(1, nx + 1)]
                        for j in range(1, ny + 1)]
        self.xedges = [hist.GetXaxis().GetBinLowEdge(i)
                             for i in range(1, nx + 2)]
        self.yedges = [hist.GetYaxis().GetBinLowEdge(i)
                             for i in range(1, ny + 2)]
        self.x      = [(self.xedges[i+1] + self.xedges[i])/2
                             for i in range(nx)]
        self.y      = [(self.yedges[i+1] + self.yedges[i])/2
                             for i in range(ny)]
        self.title  = title or hist.GetTitle()
        self.xlabel = xlabel or hist.GetXaxis().GetTitle()
        self.ylabel = ylabel or hist.GetYaxis().GetTitle()
        self.label  = label
    def _flat_content(self):
        flatcontent = []
        for row in self.content:
            flatcontent += row
        return flatcontent
    def __getitem__(self, index):
        """Return contents of indexth bin: x.__getitem__(y) <==> x[y]"""
        return self._flat_content()[index]
    def __len__(self):
        """Return the number of bins: x.__len__() <==> len(x)"""
        return len(self._flat_content())
    def __iter__(self):
        """Iterate through bins: x.__iter__() <==> iter(x)"""
        return iter(self._flat_content())
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

class Hist(object):
    """A container to hold the parameters from a ROOT histogram."""
    def __init__(self, hist, label="__nolabel__",
                 name=None, title=None, xlabel=None, ylabel=None):
        try:
            hist.GetNbinsX()
            self.__init_TH1(hist)
        except AttributeError:
            try:
                hist.GetN()
                self.__init_TGraph(hist)
            except AttributeError:
                raise TypeError("%s is not a 1D histogram or TGraph" % hist)
        self.rootclass = hist.ClassName()
        self.name = name or hist.GetName()
        self.title  = title or hist.GetTitle().split(';')[0]
        self.xlabel = xlabel or hist.GetXaxis().GetTitle()
        self.ylabel = ylabel or hist.GetYaxis().GetTitle()
        self.label  = label
    def __init_TH1(self, hist):
        self.nbins = n = hist.GetNbinsX()
        self.binlabels = process_bin_labels([hist.GetXaxis().GetBinLabel(i)
                                             for i in range(1, n + 1)])
        if self.binlabels and '' in self.binlabels:
            # Get rid of extra non-labeled bins
            self.nbins = n = self.binlabels.index('')
            self.binlabels = self.binlabels[:n]
        self.entries = hist.GetEntries()
        self.xedges = [hist.GetBinLowEdge(i) for i in range(1, n + 2)]
        self.x      = [(self.xedges[i+1] + self.xedges[i])/2 for i in range(n)]
        self.xerr   = [(self.xedges[i+1] - self.xedges[i])/2 for i in range(n)]
        self.xerr   = [self.xerr[:], self.xerr[:]]
        self.width  = [(self.xedges[i+1] - self.xedges[i])   for i in range(n)]
        self.y      = [hist.GetBinContent(i) for i in range(1, n + 1)]
        self.yerr   = [hist.GetBinError(  i) for i in range(1, n + 1)]
        self.yerr   = [self.yerr[:], self.yerr[:]]
        self.underflow = hist.GetBinContent(0)
        self.overflow  = hist.GetBinContent(self.nbins + 1)
    def __init_TGraph(self, hist):
        self.nbins = n = hist.GetN()
        self.x, self.y = [], []
        x, y = ROOT.Double(0), ROOT.Double(0)
        for i in range(n):
            hist.GetPoint(i, x, y)
            self.x.append(copy.copy(x))
            self.y.append(copy.copy(y))
        lower = [max(0, hist.GetErrorXlow(i))  for i in xrange(n)]
        upper = [max(0, hist.GetErrorXhigh(i)) for i in xrange(n)]
        self.xerr = [lower[:], upper[:]]
        lower = [max(0, hist.GetErrorYlow(i))  for i in xrange(n)]
        upper = [max(0, hist.GetErrorYhigh(i)) for i in xrange(n)]
        self.yerr = [lower[:], upper[:]]
        self.xedges = [self.x[i] - self.xerr[0][i] for i in xrange(n)]
        self.xedges.append(self.x[n - 1] + self.xerr[1][n - 1])
        self.width = [self.xedges[i + 1] - self.xedges[i] for i in range(n)]
        self.underflow, self.overflow = 0, 0
        self.binlabels = None
        self.entries = n
    def __add__(self, b):
        """Return the sum of self and b: x.__add__(y) <==> x + y"""
        c = copy.copy(self)
        for i in range(len(self)):
            c.y[i] += b.y[i]
            c.yerr[i] += b.yerr[i]
        c.overflow += b.overflow
        c.underflow += b.underflow
        return c
    def __sub__(self, b):
        """Return the difference of self and b: x.__sub__(y) <==> x - y"""
        c = copy.copy(self)
        for i in range(len(self)):
            c.y[i] -= b.y[i]
            c.yerr[i] -= b.yerr[i]
        c.overflow -= b.overflow
        c.underflow -= b.underflow
        return c
    def __div__(self, denominator):
        return self.divide(denominator)
    def __getitem__(self, index):
        """Return contents of indexth bin: x.__getitem__(y) <==> x[y]"""
        return self.y[index]
    def __setitem__(self, index, value):
        """Set contents of indexth bin: x.__setitem__(i, y) <==> x[i]=y"""
        self.y[index] = value
    def __len__(self):
        """Return the number of bins: x.__len__() <==> len(x)"""
        return self.nbins
    def __iter__(self):
        """Iterate through bins: x.__iter__() <==> iter(x)"""
        return iter(self.y)
    def min(self, threshold=None):
        """Return the y-value of the bottom tip of the lowest errorbar."""
        vals = [(yval - yerr) for yval, yerr in zip(self.y, self.yerr[0])
                if (yval - yerr) > threshold]
        if vals:
            return min(vals)
        else:
            return threshold
    def av_xerr(self):
        """Return average between the upper and lower xerr."""
        return [(self.xerr[0][i] + self.xerr[1][i]) / 2
                for i in range(self.nbins)]
    def av_yerr(self):
        """Return average between the upper and lower yerr."""
        return [(self.yerr[0][i] + self.yerr[1][i]) / 2
                for i in range(self.nbins)]
    def scale(self, factor):
        """
        Scale contents, errors, and over/underflow by the given scale factor.
        """
        self.y = [x * factor for x in self.y]
        self.yerr[0] = [x * factor for x in self.yerr[0]]
        self.yerr[1] = [x * factor for x in self.yerr[1]]
        self.overflow *= factor
        self.underflow *= factor
    def delete_bin(self, index):
        """
        Delete a the contents of a bin, sliding all the other data one bin to
        the left.  This can be useful for histograms with labeled bins.
        """
        self.nbins -= 1
        self.xedges.pop()
        self.x.pop()
        self.width.pop()
        self.y.pop(index)
        self.xerr[0].pop(index)
        self.xerr[1].pop(index)
        self.yerr[0].pop(index)
        self.yerr[1].pop(index)
        if self.binlabels:
            self.binlabels.pop(index)
    def TH1F(self, name=None):
        """Return a ROOT.TH1F object with contents of this Hist"""
        th1f = ROOT.TH1F(name or self.name, "", self.nbins,
                         array.array('f', self.xedges))
        th1f.Sumw2()
        th1f.SetTitle("%s;%s;%s" % (self.title, self.xlabel, self.ylabel))
        for i in range(self.nbins):
            th1f.SetBinContent(i + 1, self.y[i])
            try:
                th1f.SetBinError(i + 1, (self.yerr[0][i] + self.yerr[1][i]) / 2)
            except TypeError:
                th1f.SetBinError(i + 1, self.yerr[i])
            if self.binlabels:
                th1f.GetXaxis().SetBinLabel(i + 1, self.binlabels[i])
        th1f.SetBinContent(0, self.underflow)
        th1f.SetBinContent(self.nbins + 2, self.overflow)
        th1f.SetEntries(self.entries)
        return th1f
    def TGraph(self, name=None):
        """Return a ROOT.TGraphAsymmErrors object with contents of this Hist"""
        x = array.array('f', self.x)
        y = array.array('f', self.y)
        xl = array.array('f', self.xerr[0])
        xh = array.array('f', self.xerr[1])
        yl = array.array('f', self.yerr[0])
        yh = array.array('f', self.yerr[1])
        tgraph = ROOT.TGraphAsymmErrors(self.nbins, x, y, xl, xh, yl, yh)
        tgraph.SetName(name or self.name)
        tgraph.SetTitle('%s;%s;%s' % (self.title, self.xlabel, self.ylabel))
        return tgraph
    def divide(self, denominator):
        """
        Return the simple quotient with errors added in quadrature.

        This function is called by the division operator:
            hist3 = hist1.divide_wilson(hist2) <--> hist3 = hist1 / hist2
        """
        quotient = copy.deepcopy(self)
        num_yerr = self.av_yerr()
        den_yerr = denominator.av_yerr()
        quotient.yerr = [0. for i in range(self.nbins)]
        for i in range(self.nbins):
            if denominator[i] == 0 or self[i] == 0:
                quotient.y[i] = 0.
            else:
                quotient.y[i] = self[i] / denominator[i]
                quotient.yerr[i] = quotient[i]
                quotient.yerr[i] *= math.sqrt((num_yerr[i] / self[i]) ** 2 +
                                       (den_yerr[i] / denominator[i]) ** 2)
            if quotient.yerr[i] > quotient[i]:
                quotient.yerr[i] = quotient[i]
        quotient.yerr = [quotient.yerr, quotient.yerr]
        return quotient
    def divide_wilson(self, denominator):
        """Return an efficiency plot with Wilson score interval errors."""
        eff, upper_err, lower_err = wilson_interval(self.y, denominator.y)
        quotient = copy.deepcopy(self)
        quotient.y = eff
        quotient.yerr = [lower_err, upper_err]
        return quotient

class HistStack(object):
    """
    A container to hold Hist objects for plotting together.

    When plotting, the title and the x and y labels of the last Hist added
    will be used unless specified otherwise in the constructor.
    """
    def __init__(self, hists=None, title=None, xlabel=None, ylabel=None):
        self.hists  = []
        self.kwargs = []
        self.title  = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        if hists:
            for hist in hists:
                self.add(hist)
    def __getitem__(self, index):
        """Return indexth hist: x.__getitem__(y) <==> x[y]"""
        return self.hists[index]
    def __setitem__(self, index, value):
        """Replace indexth hist with value: x.__setitem__(i, y) <==> x[i]=y"""
        self.hists[index] = value
    def __len__(self):
        """Return the number of hists in the stack: x.__len__() <==> len(x)"""
        return len(self.hists)
    def __iter__(self):
        """Iterate through hists in the stack: x.__iter__() <==> iter(x)"""
        return iter(self.hists)
    def max(self):
        """Return the value of the highest bin of all hists in the stack."""
        maxes = [max(x) for x in self.hists]
        try:
            return max(maxes)
        except ValueError:
            return 0
    def stackmax(self):
        """Return the value of the highest bin in the addition of all hists."""
        try:
            return max([sum([h[i] for h in self.hists])
                       for i in range(self.hists[0].nbins)])
        except:
            print [h.nbins for h in self.hists]
    def scale(self, factor):
        """Scale all Hists by factor."""
        for hist in self.hists:
            hist.scale(factor)
    def min(self, threshold=None):
        """
        Return the value of the lowest bin of all hists in the stack.

        If threshold is specified, only values above the threshold will be
        considered.
        """
        mins = [x.min(threshold) for x in self.hists]
        return min(mins)
    def add(self, hist, **kwargs):
        """
        Add a Hist object to this stack.

        Any additional keyword arguments will be added to just this Hist
        when the stack is plotted.
        """
        if "label" in kwargs:
            hist.label = kwargs['label']
            del kwargs['label']
        self.hists.append(hist)
        self.kwargs.append(kwargs)


################ Define functions and classes for navigating within ROOT

class RootFile(object):
    """A wrapper for TFiles, allowing easier access to methods."""
    def __init__(self, filename, name=None):
        self.filename = filename
        self.name = name or filename[:-5]
        self.file = ROOT.TFile(filename, 'read')
        if self.file.IsZombie():
            raise ValueError("Error opening %s" % filename)
    def cd(self, directory=''):
        """Make directory the current directory."""
        self.file.cd(directory)
    def get(self, object_name, path=None, type1D=Hist, type2D=Hist2D):
        """Return a Hist object from the given path within this file."""
        if not path:
            path = os.path.dirname(object_name)
            object_name = os.path.basename(object_name)
        try:
            roothist = self.file.GetDirectory(path).Get(object_name)
        except ReferenceError, e:
            raise ReferenceError(e)
        try:
            return type2D(roothist)
        except TypeError:
            return type1D(roothist)

def ls(directory=None):
    """Return a python list of ROOT object names from the given directory."""
    if directory == None:
        keys = ROOT.gDirectory.GetListOfKeys()
    else:
        keys = ROOT.gDirectory.GetDirectory(directory).GetListOfKeys()
    key = keys[0]
    names = []
    while key:
        obj = key.ReadObj()
        key = keys.After(key)
        names.append(obj.GetName())
    return names

def pwd():
    """Return ROOT's present working directory."""
    return ROOT.gDirectory.GetPath()

def get(object_name):
    """Return a Hist object with the given name."""
    return Hist(ROOT.gDirectory.Get(object_name))


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

def process_bin_labels(binlabels):
    has_labels = False
    for binlabel in binlabels:
        if binlabel:
            has_labels = True
    if has_labels:
        return binlabels
    else:
        return None

def wilson_interval(numerator_array, denominator_array):
    eff, upper_err, lower_err = [], [], []
    for n, d in zip(numerator_array, denominator_array):
        try:
            p = float(n) / d
            s = math.sqrt(p * (1 - p) / d + 1 / (4 * d * d))
            t = p + 1 / (2 * d)
            eff.append(p)
            upper_err.append(-p + 1/(1 + 1/d) * (t + s))
            lower_err.append(+p - 1/(1 + 1/d) * (t - s))
        except ZeroDivisionError:
            eff.append(0)
            upper_err.append(0)
            lower_err.append(0)
    return eff, upper_err, lower_err

def find_num_processors():
    import os
    try:
        num_processors = os.sysconf('SC_NPROCESSORS_ONLN')
    except:
        try:
            num_processors = os.environ['NUMBER_OF_PROCESSORS']
        except:
            num_processors = 1
    return num_processors
