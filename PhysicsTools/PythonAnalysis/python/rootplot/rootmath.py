"""
rootmath description
"""
from __future__ import absolute_import

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


##############################################################################
######## Import python libraries #############################################

import sys
import shutil
import math
import os
import re
import tempfile
import copy
import fnmatch
from . import argparse
from os.path import join as joined
from .utilities import rootglob, loadROOT

ROOT = loadROOT()

##############################################################################
######## Define globals ######################################################

from .version import __version__          # version number


##############################################################################
######## Classes #############################################################

class Target(object):
    """Description."""
    def __init__(self, filename, path='', scale=1., scale_error=None):
        self.filename = filename
        self.path = path
        self.scale = scale
        self.scale_error = scale_error
    def __repr__(self):
        return "%s:%s:%f" % (self.filename, self.path, self.scale)

def newadd(outfile, targets, dest_path=""):
    """Description."""
    if allsame([x.filename for x in targets]):
        f = ROOT.TFile(targets[0].filename, 'read')
        paths = [x.path for x in targets]
        scales = [x.scale for x in targets]
        scale_errors = [x.scale_error for x in targets]
        if f.GetDirectory(paths[0]):
            destdir = pathdiff2(paths)    # What does this do?
            for h in [os.path.basename(x) for x in
                      rootglob(f, paths[0] + '/*')]:
                hists = [f.GetDirectory(x).Get(h) for x in paths]
                if not alltrue([x and x.InheritsFrom('TH1') for x in hists]):
                    continue
                dest = joined(destdir, h)
                add(outfile, dest, hists, scales, dest_path, scale_errors=scale_errors)
        else:
            hists = [f.Get(x) for x in paths]
            if alltrue([x and x.InheritsFrom('TH1') for x in hists]):
                dest = pathdiff2(paths)
                add(outfile, dest, hists, scales, scale_errors=scale_errors)
    else:
        dict_targets = {}  # Stores paths and scales, key = filename
        dict_tfiles = {}   # Stores map from filenames to Root.TFile() objects
        for target in targets:
            dict_targets.setdefault(target.filename, []).append((target.path, target.scale))
            if (target.filename not in dict_tfiles):
                # Only open root files once
                dict_tfiles[target.filename] = ROOT.TFile(target.filename, 'read')
        # dict_targets now a dictionary, with keys the filenames, example:
        # {'fileA.root': [('path0',scale0), ('path1', scale1)],
        #  'fileB.root': [('path3', scale3)]}
        f = ROOT.TFile(targets[0].filename, 'read')
        if f.GetDirectory(targets[0].path):
            # Create list of histograms to get
            destdir = '/'               # should probably use pathdiff2 somehow
            histnames = [os.path.basename(x) for x in
                         rootglob(f, targets[0].path + '/*')]
            f.Close()
            # For each histogram name found, grab it from
            # every file & path
            for histname in histnames:
                hists = []
                scales = []
                for filename in dict_targets:
                    tfile_cur = dict_tfiles[filename]
                    for path, scale in dict_targets[filename]:
                        hists.append(tfile_cur.GetDirectory(path).Get(histname))
                        scales.append(scale)
                        #print "%s:%s:%s:%f" % (filename, path, histname, scale)
                if not alltrue([x and x.InheritsFrom('TH1') for x in hists]):
                    continue
                dest = joined(destdir, histname)
                add(outfile, dest, hists, scales, dest_path)
        else:
            print "Code not written yet to add histograms from multiple files"
            return
        return


##############################################################################
######## Implementation ######################################################

def walk_rootfile(rootfile, path=''):
    #### Yield (path, folders, objects) for each directory under path.
    keys = rootfile.GetDirectory(path).GetListOfKeys()
    folders, objects = [], []
    for key in keys:
        name = key.GetName()
        classname = key.GetClassName()
        newpath = joined(path, name)
        dimension = 0
        if 'TDirectory' in classname:
            folders.append(name)
        else:
            objects.append(name)
    yield path, folders, objects
    for folder in folders:
        for x in walk_rootfile(rootfile, joined(path, folder)):
            yield x

def allsame(iterable):
    for element in iterable:
        if element != iterable[0]:
            return False
    return True

def alltrue(iterable):
    for element in iterable:
        if element != True:
            return False
    return True

def pathdiff(paths, joiner):
    """
    Return the appropriate destination for an object.
    
    In all cases, the result will be placed in the deepest directory shared by
    all paths.  If the histogram names are the same, the result will be named
    based on the first directories that they do not share.  Otherwise, the 
    result will be named based on the names of the other histograms.

    >>> pathdiff(['/dirA/dirB/dirX/hist', '/dirA/dirB/dirY/hist'], '_div_')
    '/dirA/dirB/dirX_div_dirY'
    >>> pathdiff(['/dirA/hist1', '/dirA/hist2', '/dirA/hist3'], '_plus_')
    '/dirA/hist1_plus_hist2_plus_hist3'
    >>> pathdiff(['/hist1', '/dirA/hist2'], '_minus_')
    '/hist1_minus_hist2'
    """
    paths = [x.split('/') for x in paths]
    dest = '/'
    for i in range(len(paths[0])):
        if allsame([p[i] for p in paths]):
            dest = joined(dest, paths[0][i])
        else:
            break
    name = joiner.join([p[-1] for p in paths])
    if allsame([p[-1] for p in paths]):
        for i in range(len(paths[0])):
            if not allsame([p[i] for p in paths]):
                name = joiner.join([p[i] for p in paths])
    return joined(dest, name)

def pathdiff2(paths, joiner='__', truncate=False):
    """
    Placeholder.
    """
    paths = [x.split('/') for x in paths]
    commonbeginning = ''
    for i in range(len(paths[0])):
        if allsame([p[i] for p in paths]):
            commonbeginning = joined(commonbeginning, paths[0][i])
        else:
            break
    commonending = ''
    for i in range(-1, -1 * len(paths[0]), -1):
        if allsame([p[i] for p in paths]):
            commonending = joined(paths[0][i], commonending)
        else:
            break
    #return commonbeginning, commonending
    if truncate:
        return commonending
    else:
        return joined(commonbeginning, commonending)

def pathdiff3(paths, joiner='__'):
    """
    Return the appropriate destination for an object.
    
    If the final objects in each path match, then the return value will be the
    matching part of the paths.  Otherwise, the output path will simply be those
    names joined together with *joiner*.  See the examples below.
    
    >>> pathdiff3(['/dirA/dirX/hist', '/dirA/dirY/hist'])
    '/hist'
    >>> pathdiff3(['/dirA/dirX/dirB/hist', '/dirA/dirY/dirB/hist'])
    '/dirB/hist'
    >>> pathdiff3(['/dirA/hist1', '/dirA/hist2', '/dirA/hist3'], '_plus_')
    '/hist1_plus_hist2_plus_hist3'
    >>> pathdiff3(['/hist1', '/dirA/hist2'], '_div_')
    '/hist1_div_hist2'
    """
    paths = [x.split('/') for x in paths]
    if allsame([x[-1] for x in paths]):
        dest = paths[0][-1]
        for i in range(-2, min([len(x) for x in paths]) * -1, -1):
            if allsame([p[i] for p in paths]):
                dest = joined(paths[0][i], dest)
            else:
                break
        return '/' + dest
    else:
        return '/' + joiner.join([x[-1] for x in paths])

def operator_func(fn):
    def newfunc(outfile, dest, hists, scales=None, dest_path="", scale_errors=None):
        outfile.cd()
        for d in os.path.dirname(dest).split('/'):
            if not ROOT.gDirectory.GetDirectory(d):
                ROOT.gDirectory.mkdir(d)
            ROOT.gDirectory.cd(d)
        fn(outfile, dest, hists, scales, dest_path, scale_errors)
    return newfunc

def scale_with_error(hist, scale, scale_error=None):
    '''Scale a histogram by a scale factor that has an error.
    This takes into account the scale error to set new error bars.'''
    hist_new = hist.Clone()
    if scale_error:
        for i in range(hist_new.GetNbinsX()+2):
            hist_new.SetBinContent(i, scale)
            hist_new.SetBinError(i, scale_error)
        hist_new.Multiply(hist)
    else:
        hist_new.Scale(scale)
    return hist_new

@operator_func
def add(outfile, dest, hists, scales=None, dest_path="", scale_errors=None):
    if not scales:
        scales = [1. for i in range(len(hists))]
    if not scale_errors:
        scale_errors = [None for i in range(len(hists))]
    sumhist = hists[0].Clone(os.path.basename(dest))
    sumhist = scale_with_error(sumhist, scales[0], scale_errors[0])
    #sumhist.Scale(scales[0])
    for i in range(1,len(hists)):
        sumhist.Add(scale_with_error(hists[i], scales[i], scale_errors[i]))
        #sumhist.Add(hists[i], scales[i])
    if dest_path:
        outfile.cd()
        if not ROOT.gDirectory.GetDirectory(dest_path):
            ROOT.gDirectory.mkdir(dest_path)
        ROOT.gDirectory.cd(dest_path)
    sumhist.Write()
    ROOT.gDirectory.cd("/")

@operator_func
def subtract(outfile, dest, hists):
    diffhist = hists[0].Clone(os.path.basename(dest))
    for hist in hists[1:]:
        diffhist.Add(hist, -1)
    diffhist.Write()

@operator_func
def divide(outfile, dest, numer, denom):
    quotient = numer.Clone(os.path.basename(dest))
    quotient.Divide(numer, denom)
    quotient.Write()

@operator_func
def bayes_divide(outfile, dest, numer, denom):
    quotient = ROOT.TGraphAsymmErrors()
    quotient.SetName(os.path.basename(dest))
    quotient.BayesDivide(numer, denom)
    quotient.Write()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str, nargs='+',
                       help='root files to process')
    parser.add_argument('--dirs', type=str, nargs='+', default=['/'],
                        help='target directories in the root files; paths to '
                        'histograms will be relative to these')
    parser.add_argument('--add', default=[], action='append', nargs='*',
                        help='a list of directories or histograms to add')
    parser.add_argument('--subtract', default=[], action='append', nargs='*',
                        help='a list of directories or histograms to subtract')
    parser.add_argument('--divide', default=[], action='append', nargs='*',
                        help='2 directories or histograms to divide')
    parser.add_argument('--bayes-divide', default=[], action='append', nargs='*',
                        help='2 directories or histograms from which to make '
                        'an efficiency plot')
    args = parser.parse_args()
    separators = {'add' : '_plus_',
                  'subtract' : '_minus_',
                  'divide' : '_div_',
                  'bayes_divide' : '_eff_'}

    files = [ROOT.TFile(x, 'read') for x in args.filenames]
    outfile = ROOT.TFile('out.root', 'recreate')
    dirs = []
    for d in args.dirs:
        dirs += rootglob(files[0], d)

    if len(files) == 1:
        f = files[0]
        for thisdir in dirs:
            for operation_type, separator in separators.items():
                for arg_set in getattr(args, operation_type):
                    paths = [joined(thisdir, x) for x in arg_set]
                    if f.GetDirectory(paths[0]):
                        destdir = pathdiff(paths, separator)
                        for target in [os.path.basename(x) for x in
                                       rootglob(f, paths[0] + '/*')]:
                            hists = [f.GetDirectory(x).Get(target)
                                     for x in paths]
                            if not alltrue([x and x.InheritsFrom('TH1')
                                            for x in hists]):
                                continue
                            dest = joined(destdir, target)
                            math_func = globals()[operation_type]
                            math_func(outfile, dest, hists)
                    else:
                        hists = [f.GetDirectory(thisdir).Get(x) for x in paths]
                        if not alltrue([x and x.InheritsFrom('TH1') 
                                        for x in hists]):
                            continue
                        dest = pathdiff(paths, separator)
                        math_func = globals()[operation_type]
                        math_func(outfile, dest, hists)
    else:
        for operation_type, separator in separators.items():
            arg_sets = getattr(args, operation_type)
            if arg_sets and arg_sets != [[]]:
                raise ValueError("No arguments to --%s allowed when multiple "
                                 "files are specified" % operation_type)
            elif arg_sets:
                if 'divide' in operation_type and len(files) != 2:
                    raise ValueError("Exactly 2 files are expected with --%s; "
                                     "%i given" % (operation_type, len(files)))
                for path, folders, objects in walk_rootfile(files[0]):
                    for obj in objects:
                        hists = [x.GetDirectory(path).Get(obj) for x in files]
                        if not alltrue([x and x.InheritsFrom('TH1') 
                                        for x in hists]):
                            continue
                        math_func = globals()[operation_type]
                        math_func(outfile, joined(path, obj), hists)

    outfile.Close()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
