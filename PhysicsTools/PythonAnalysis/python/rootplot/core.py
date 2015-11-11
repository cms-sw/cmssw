"""
An API and a CLI for quickly building complex figures.
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

usage="""\
Usage: %prog [config.py] targets [options]

Targets may be either multiple root files to compare or a single root file 
followed by multiple histograms or folders to compare.  For example:
    %prog fileA.root fileB.root fileC.root
    %prog file.root dirA dirB dirC
    %prog file.root dirA/hist1 dirA/hist2

Full documentation is available at: 
    http://packages.python.org/rootplot/"""

##############################################################################
######## Import python libraries #############################################

import sys
import optparse
import shutil
import math
import os
import re
import tempfile
import copy
from os.path import join as joined


##############################################################################
######## Import ROOT and rootplot libraries ##################################

from .utilities import RootFile, Hist, Hist2D, HistStack
from .utilities import find_num_processors, loadROOT

argstring = ' '.join(sys.argv)
## Use ROOT's batch mode, unless outputting to C macros, since there is
## a bug in pyROOT that fails to export colors in batch mode
batch = (not re.search('--ext[ =]*C', argstring) and
         not re.search('-e[ ]*C', argstring))
ROOT = loadROOT(batch=batch)


##############################################################################
######## Define globals ######################################################

from .version import __version__          # version number
prog = os.path.basename(sys.argv[0])     # rootplot or rootplotmpl
use_mpl = False                          # set in plotmpl or rootplotmpl
global_opts = ['filenames', 'targets', 'debug', 'path', 'processors', 
               'merge', 'noclean', 'output', 'numbering', 'html_template',
               'ncolumns_html']
try:
    import multiprocessing
    use_multiprocessing = True
except ImportError:
    use_multiprocessing = False


##############################################################################
######## Classes #############################################################

class Options(dict):
    def __init__(self, options, arguments, scope='global'):
        for opt in dir(options):
            value = getattr(options, opt)
            if (not opt.startswith('__') and
                type(value) in [int, float, str, bool, type(None)]):
                self[opt] = value
        self.filenames = [x for x in arguments if x.endswith('.root')]
        self.configs   = [x for x in arguments if x.endswith('.py')]
        self.targets   = [x for x in arguments if not (x.endswith('.py') or
                                                       x.endswith('.root'))]
        self.process_configs(scope)
    def __setattr__(self, key, value):
        self[key] = value
    def __getattr__(self, key):
        return self[key]
    def clean_targets(self):
        for i in range(len(self.targets)):
            if self.targets[i][-1] == '/':
                self.targets[i] = self.targets[i][:-1]
    def arguments(self):
        return self.filenames + self.targets + self.configs
    def kwarg_list(self):
        diffs = {}
        defaults = parse_arguments([])
        for key, value in self.items():
            if (key not in ['filenames', 'targets', 'configs'] and
                defaults[key] != value):
                diffs[key] = value
        return diffs
    def append_from_package(self, package, scope):
        for attribute in dir(package):
            if '__' not in attribute:
                if ((scope == 'global' and attribute in global_opts) or 
                    (scope == 'plot' and attribute not in global_opts)):
                    value = getattr(package, attribute)
                    self[attribute] = value
    def process_configs(self, scope):
        #### Load variables from configs; scope is 'global' or 'plot'
        configdir = tempfile.mkdtemp()
        sys.path.insert(0, '')
        sys.path.insert(0, configdir)
        write_to_file(config_string(), 
                      joined(configdir, 'default_config.py'))
        configs = ['default_config.py']
        for i, c in enumerate(self.configs):
            shutil.copy(c, joined(configdir, 'rpconfig%i.py' % i))
            configs.append('rpconfig%i.py' % i)
        rc_name = use_mpl and 'rootplotmplrc' or 'rootplotrc'
        rc_path = os.path.expanduser('~/.%s' % rc_name)
        if os.path.exists(rc_path):
            print "Using styles and options from ~/.%s" % rc_name
            shutil.copy(rc_path, joined(configdir, '%s.py' % rc_name))
            configs.insert(1, '%s.py' % rc_name)
        for f in configs:
            myconfig = __import__(f[:-3])
            self.append_from_package(myconfig, scope)
        self.clean_targets()
        shutil.rmtree(configdir)


##############################################################################
######## Templates ###########################################################

config_template=r"""
import ROOT         # allows access to ROOT colors (e.g. ROOT.kRed)

##############################################################################
######## About Config Files ##################################################

## This file can be generated by running '%prog --config'

## Options are loaded in the following order:
##   1. from the command line
##   2. from the default configuration file
##   3. from ~/.%progrc
##   4. from configuration files specified on the command-line
## This leads to two major points worth understanding:
##   1. you may delete any lines you like in this file and they will still
##      be loaded correctly from the default
##   2. values specified here will superceed the same options from the
##      command-line
## Therefore, you could set, for example, 'xerr = True' in this file,
## and x-errorbars will always be drawn, regardless of whether '--xerr' is
## given on the command-line or not.  You can do this with any of the command-
## line options, but note that dashes are translated to underscores, so
## '--ratio-split=1' becomes 'ratio_split = 1'.

## Most global style options like default line widths can be set through
root::## a rootlogon.C, as described at:
root::##    http://root.cern.ch/drupal/content/how-create-or-modify-style
mpl:::## a matplotlibrc, as described at:
mpl:::##    http://matplotlib.sourceforge.net/users/customizing.html

##############################################################################
######## Specifying Files and Targets ########################################

## You can specify the files to run on through the 'filenames' variable rather
## than entering them at the command-line, for example:
## filenames = ['histTTbar.root', 'histZmumu.root']

## Likewise, you can specify target histograms or directories here rather than 
## on the command-line, for example:
## targets = ['barrel/15to20', 'barrel/20to30']

## You might also want to specify fancy labels for the legend here rather 
## than on the command-line:
root::## legend_entries = [r'#bar{t}t', r'Z#rightarrow#mu#mu']
mpl:::## legend_entries = [r'$\bar{t}t$', r'$Z\rightarrow\mu\mu$']

##############################################################################
######## Different Options for Different Targets #############################

## Leave these lists empty to have them automatically filled according to the
## command-line options.  Any list that is filled must be at least as long
## as the number of targets or it will throw an error.

line_colors = []                # normally filled by options.colors
fill_colors = []                # normally filled by options.colors
marker_colors = []              # normally filled by options.colors
mpl:::errorbar_colors = []      # color for bars around the central value
root::marker_sizes = []         # in pixels
mpl:::marker_sizes = []         # in points
root::line_styles = []          # 1 (solid), 2 (dashed), 4 (dashdot), 3 (dotted), ...
mpl:::line_styles = []          # 'solid', 'dashed', 'dashdot', 'dotted'
root::fill_styles = []          # 0 (hollow), 1001 (solid), 2001 (hatched), ...
mpl:::fill_styles = []          # None, '/', '\', '|', '-', '+', 'x', 'o', 'O', ...
root::draw_commands = []        # a TH1::Draw option, include 'stack' to make stacked
mpl:::plot_styles = []          # 'bar', 'hist', 'errorbar', 'stack'
mpl:::alphas = []               # transparencies for fills (value from 0 to 1)

##############################################################################
######## Global Style Options ################################################

## Colors can be specified as (r, g, b) tuples (with range 0. to 1. or range
root::## 0 to 255), or ROOT color constants (ROOT.kBlue or 600)
mpl:::## 0 to 255), ROOT color constants (ROOT.kBlue or 600), or any matplotlib
mpl:::## color specification (names like 'blue' or 'b')

colors = [
    ## a default set of contrasting colors the author happens to like
    ( 82, 124, 219), # blue
    (212,  58, 143), # red
    (231, 139,  77), # orange
    (145,  83, 207), # purple
    (114, 173, 117), # green
    ( 67,  77,  83), # dark grey
    ]

## Used when --marker_styles is specified; more info available at:
root::## http://root.cern.ch/root/html/TAttMarker.html
mpl:::## http://matplotlib.sourceforge.net/api/
mpl:::##        artist_api.html#matplotlib.lines.Line2D.set_marker
marker_styles = [
mpl:::    'o', 's', '^', 'x', '*', 'D', 'h', '1'
root::     4, # circle
root::    25, # square
root::    26, # triangle
root::     5, # x
root::    30, # five-pointed star
root::    27, # diamond
root::    28, # cross
root::     3, # asterisk
    ]

#### Styles for --data
root::data_linestyle = 1
mpl:::data_linestyle = 'solid'
data_color = (0,0,0)      # black
mc_color = (50, 150, 150) # used when there are exactly 2 targets; set to
                          # None to pick up the normal color
root::data_marker = 4           # marker style (circle)
mpl:::data_marker = 'o'         # marker style

#### Settings for --ratio-split or --efficiency-split
ratio_max  = None
ratio_min  = None
ratio_logy = False
ratio_fraction = 0.3  # Fraction of the canvas that bottom plot occupies
ratio_label = 'Ratio to %(ratio_file)s' # Label for the bottom plot
efficiency_label = 'Efficiency vs. %(ratio_file)s'

#### Titles produced by --area-normalize and --normalize
area_normalized_title = 'Fraction of Events in Bin'
target_normalized_title = 'Events Normalized to %(norm_file)s'

#### Overflow and underflow text labels
overflow_text = ' Overflow'
underflow_text = ' Underflow'
mpl:::overflow_size = 'small'
mpl:::overflow_alpha = 0.5

#### Define how much headroom to add to the plot
top_padding_factor = 1.2
top_padding_factor_log = 5.    # used when --logy is set

#### Plotting options based on histogram names
## Apply options to histograms whose names match regular expressions
## The tuples are of the form (option_name, value_to_apply, list_of_regexs)
## ex: to rebin by 4 all histograms containing 'pt' or starting with 'eta':
##    ('rebin', 4, ['.*pt.*', 'eta.*'])
options_by_histname = [
    ('area_normalize', True, []),
                       ]

root::#### Legend
root::legend_width = 0.38        # Fraction of canvas width
root::legend_entry_height = 0.05 # Fraction of canvas height
root::max_legend_height = 0.4    # Fraction of canvas height
root::legend_left_bound = 0.20   # For left justification
root::legend_right_bound = 0.95  # For right justification
root::legend_upper_bound = 0.91  # For top justification
root::legend_lower_bound = 0.15  # For bottom justification
root::legend_codes = { 1 : 'upper right',
root::                 2 : 'upper left',
root::                 3 : 'lower left',
root::                 4 : 'lower right',
root::                 5 : 'right',
root::                 6 : 'center left',
root::                 7 : 'center right',
root::                 8 : 'lower center',
root::                 9 : 'upper center',
root::                10 : 'center',
root::                }
root::
root::#### Page numbers
root::numbering_size_root = 0.03  # Fraction of canvas width
root::numbering_align_root = 33   # Right-top adjusted
root::numbering_x_root = 0.97     # Fraction of canvas width
root::numbering_y_root = 0.985    # Fraction of canvas height
root::
root::#### Draw style for TGraph
root::draw_graph = 'ap'
root::
root::#### This code snippet will be executed after the histograms have all
root::#### been drawn, allowing you to add decorations to the canvas
root::decoration_root = '''
root::## Draw a line to indicate a cut
root::#line = ROOT.TLine(5.,0.,5.,9.e9)
root::#line.Draw()
root::## Add a caption
root::#tt = ROOT.TText()
root::#tt.DrawTextNDC(0.6, 0.15, "CMS Preliminary")
root::'''
mpl:::#### Legend
mpl:::#### These options will override legend_location, allowing more precise control
mpl:::## Upper right corner of legend in figure coordinates
mpl:::legend_figure_bbox = None    # [1.0, 1.0] for legend outside the axes
mpl:::## Upper right corner of legend in axes coordinates
mpl:::legend_axes_bbox = None
mpl:::
mpl:::#### Page numbers
mpl:::numbering_size_mpl = 'small'
mpl:::numbering_ha_mpl = 'right'
mpl:::numbering_va_mpl = 'top'
mpl:::numbering_x_mpl = 0.98       # Fraction of canvas width
mpl:::numbering_y_mpl = 0.98       # Fraction of canvas height
mpl:::
mpl:::#### Rotation for text x-axis labels
mpl:::xlabel_rotation = -15
mpl:::xlabel_alignment = 'left'
mpl:::xlabel_alignmenth = 'bottom' # For barh
mpl:::
mpl:::#### Convert ROOT symbols to proper LaTeX, for matplotlib plotting
mpl:::## By default, matplotlib renders only symbols between $'s as TeX, but if
mpl:::## you enable the 'text.usetex' matplotlibrc setting, then everything is handled
mpl:::## by the LaTeX engine on your system, in which case you can go wild with TeX.
mpl:::
mpl:::## ROOT-type strings on left get replaced with LaTeX strings on the right
mpl:::replace = [
mpl:::    # some defaults that should work for most cases
mpl:::    (' pt '    , r' $p_\mathrm{T}$ '),
mpl:::    ('pT '     , r'$p_\mathrm{T}$ '),
mpl:::    (' pT'     , r' $p_\mathrm{T}$'),
mpl:::    ('p_{T}'   , r'$p_\mathrm{T}$'),
mpl:::    ('E_{T}'   , r'$E_\mathrm{T}$'),
mpl:::    ('#eta'    , r'$\eta$'),
mpl:::    ('#phi'    , r'$\phi$'),
mpl:::    ('fb^{-1}' , r'$\mathrm{fb}^{-1}$'),
mpl:::    ('pb^{-1}' , r'$\mathrm{pb}^{-1}$'),
mpl:::    ('<'       , r'$<$'),
mpl:::    ('>'       , r'$>$'),
mpl:::    ('#'       , r''),
mpl:::    ]
mpl:::
mpl:::## If you include 'use_regexp' as the first item, the patterns to be replaced
mpl:::## will function as regular expressions using python's re module rather than
mpl:::## as simple text.  The example below turn's ROOT's superscript and subscript
mpl:::## syntax into LaTeX:
mpl:::
mpl:::## replace = [
mpl:::##     ('use_regexp', True),
mpl:::##     (r'\^\{(.*)\}', r'$^{\1}$'),
mpl:::##     (r'\_\{(.*)\}', r'$_{\1}$'),
mpl:::## ]
mpl:::
mpl:::#### A function that will be executed after all histograms have been drawn.
mpl:::#### It can be used to add extra decorations to your figure.
mpl:::def decoration_mpl(figure, axeses, path, options, hists):
mpl:::    #### Draw a line to indicate a cut
mpl:::    ## axeses[0].axvline(5., color='red', linestyle='--')
mpl:::    #### Add a caption
mpl:::    ## figure.text(0.6, 0.15, "CMS Preliminary")
mpl:::    return

##############################################################################
######## HTML Output #########################################################

#### Number of columns for images in HTML output
ncolumns_html = 2

#### Provide a template for the html index files
html_template=r'''
    <html>
    <head>
    <link rel='shortcut icon' href='http://packages.python.org/rootplot/_static/rootplot.ico'>
    <link href='http://fonts.googleapis.com/css?family=Yanone+Kaffeesatz:bold' rel='stylesheet' type='text/css'>
    <style type='text/css'>
        body { padding: 10px; font-family:Arial, Helvetica, sans-serif;
               font-size:15px; color:#FFF; font-size: large; 
               background-image: url(
                   'http://packages.python.org/rootplot/_static/tile.jpg');}
        img    { border: solid black 1px; margin:10px; }
        object { border: solid black 1px; margin:10px; }
        h1   { text-shadow: 2px 2px 2px #000;
               font-size:105px; color:#fff; border-bottom: solid black 1px;
               font-size: 300%%; font-family: 'Yanone Kaffeesatz'}
        a, a:active, a:visited {
               color:#FADA00; text-decoration:none; }
        a:hover{ color:#FFFF00; text-decoration:none;
                 text-shadow: 0px 0px 5px #fff; }
    </style>
    <title>%(path)s</title>
    </head>
    <body>
    <a style="" href="http://packages.python.org/rootplot/"><img style="position: absolute; top:10 px; right: 10px; border: 0px" src="http://packages.python.org/rootplot/_static/rootplot-logo.png"></a>
    <h1>Navigation</h1>
      %(back_nav)s
      <ul>
          %(forward_nav)s
      </ul>
    <h1>Images</h1>
    %(plots)s
    <p style='font-size: x-small; text-align: center;'>
      <a href='http://www.greepit.com/resume-template/resume.htm'>
        Based on a template by Sarfraz Shoukat</a></p>
    </body>
    </html>
'''
"""

multi_call_template = '''
calls.append("""
%s
%s
""")
'''

allplots_template = '''
## This file contains all the necessary calls to the rootplot API to produce
## the same set of plots that were created from the command-line.

## You can use this file to intercept the objects and manipulate them before
## the figure is saved, making any custom changes that are not possible from
## the command-line.

## 'objects' is a python dictionary containing all the elements used in the
## plot, including 'hists', 'legend', etc.
##   ex: objects['hists'] returns a list of histograms

try:
  ## the normal way to import rootplot
  from rootplot import plot, plotmpl
except ImportError:
  ## special import for CMSSW installations of rootplot
  from PhysicsTools.PythonAnalysis.rootplot import plot, plotmpl

import os
os.chdir('..')  # return to the directory with the ROOT files

%(call_statements)s
'''

allplots_multi_template = '''
## This file is the same as allplots.py, except that it uses multiprocessing
## to make better use of machines with multiple cores

try:
  ## the normal way to import rootplot
  from rootplot import plot, plotmpl
  from rootplot.core import report_progress
except ImportError:
  ## special import for CMSSW installations of rootplot
  from PhysicsTools.PythonAnalysis.rootplot import plot, plotmpl
  from PhysicsTools.PythonAnalysis.rootplot.core import report_progress
import ROOT
import multiprocessing as multi

import os
os.chdir('..')  # return to the directory with the ROOT files

calls = []

%(call_statements)s

queue = multi.JoinableQueue()
qglobals = multi.Manager().Namespace()
qglobals.nfinished = 0
qglobals.ntotal = len(calls)
for call in calls:
    queue.put(call)

def qfunc(queue, qglobals):
    from Queue import Empty
    while True:
        try: mycall = queue.get(timeout=5)
        except (Empty, IOError): break
        exec(mycall)
        ROOT.gROOT.GetListOfCanvases().Clear()
        qglobals.nfinished += 1
        report_progress(qglobals.nfinished, qglobals.ntotal, 
                        '%(output)s', '%(ext)s')
        queue.task_done()

for i in range(%(processors)i):
    p = multi.Process(target=qfunc, args=(queue, qglobals))
    p.daemon = True
    p.start()
queue.join()
report_progress(len(calls), len(calls), '%(output)s', '%(ext)s')
print ''
'''

## remove leading blank lines from the templates
for key, value in globals().items():
    if 'template' in key:
        globals()[key] = value[1:]

##############################################################################
######## The Command-Line Interface ##########################################

def cli_rootplotmpl():
    """
    An application for plotting histograms from a ROOT file with |matplotlib|.

    It is invoked from the command-line as ``rootplotmpl``.
    """
    global use_mpl
    use_mpl = True
    cli_rootplot()
    
def cli_rootplot():
    """
    An application for plotting histograms from a ROOT file.

    It is invoked from the command-line as ``rootplot``.
    """
    options = parse_arguments(sys.argv[1:])
    optdiff = options.kwarg_list()
    if options.debug:
        rootplot(*options.arguments(), **optdiff)
    else:
        try:
            rootplot(*options.arguments(), **optdiff)
        except Exception as e:
            print "Error:", e
            print "For usage details, call '%s --help'" % prog
            sys.exit(1)

##############################################################################
######## The Application Programming Interface ###############################

def plotmpl(*args, **kwargs):
    """
    call signature::

      plotmpl(file1, file2, file3, ..., target, **kwargs):

    build a matplotlib figure, pulling the *target* histogram from each of the
    *files*.

    call signature::

      plotmpl(file, target1, target2, target3, ..., **kwargs):

    build a matplotlib figure, pulling all *target* histograms from *file*.

    With either of these signatures, the plot style is specified through
    *kwargs*, which can accept any of the options available to
    :mod:`rootplotmpl` at the command-line.

    Returns the tuple (*figure*, *axeses*, *stack*, *hists*, *plotpath*).
    """
    global use_mpl
    use_mpl = True
    return plot(*args, **kwargs)

def plot(*args, **kwargs):
    """
    call signature::

      plot(file1, file2, file3, ..., target, **kwargs):

    build a ROOT canvas, pulling the *target* histogram from each of the
    *files*.

    call signature::

      plotmpl(file, target1, target2, target3, ..., **kwargs):

    build a ROOT canvas, pulling all *target* histograms from *file*.

    With either of these signatures, the plot style is specified through
    *kwargs*, which can accept any of the options available to
    :mod:`rootplot` at the command-line.

    Returns the tuple (*canvas*, *pads*, *stack*, *hists*, *plotpath*).
    """
    hists, options = initialize_hists(args, kwargs)
    if use_mpl:
        return plot_hists_mpl(hists, options)
    else:
        return plot_hists_root(hists, options)

def rootplotmpl(*args, **kwargs):
    """
    call signature::

      rootplotmpl(file1, file2, file3, ..., **kwargs):

    build ROOT canvases from corresponding histograms in each of the *files*.

    call signature::

      rootplotmpl(file, folder1, folder2, folder3, ..., **kwargs):

    build ROOT canvases from corresponding histograms in each of the *folders*
    in *file*.

    call signature::

      rootplotmpl(file, target1, target2, target3, ..., **kwargs):

    build a ROOT canvas from each of the *targets* in *file*.

    With any of these call signatures, images are generated in an output
    directory along with a script with the necessary calls to :func:`plotmpl`
    to reproduce each of the canvases.  The plot style is specified through
    *kwargs*, which can accept any of the options available to
    :mod:`rootplotmpl` at the command-line.
    """
    global use_mpl
    use_mpl = True
    rootplot(args, kwargs)

def rootplot(*args, **kwargs):
    """
    call signature::

      rootplot(file1, file2, file3, ..., **kwargs):

    build ROOT canvases from corresponding histograms in each of the *files*.

    call signature::

      rootplot(file, folder1, folder2, folder3, ..., **kwargs):

    build ROOT canvases from corresponding histograms in each of the *folders*
    in *file*.

    call signature::

      rootplot(file, target1, target2, target3, ..., **kwargs):

    build a ROOT canvas from each of the *targets* in *file*.

    With any of these call signatures, images are generated in an output
    directory along with a script with the necessary calls to :func:`plot`
    to reproduce each of the canvases.  The plot style is specified through
    *kwargs*, which can accept any of the options available to
    :mod:`rootplot` at the command-line.
    """
    if 'config' in kwargs:
        write_config()
    options = fill_options(args, kwargs, scope='global')
    nfiles = len(options.filenames)
    ntargets = len(options.targets)
    if nfiles < 1:
        raise TypeError("%s takes at least 1 filename argument (0 given)" %
                        prog)
    elif ntargets > 0 and nfiles > 1:
        raise TypeError("%s cannot accept targets (%i given) when "
                        "multiple files are specified (%i given)" %
                        (prog, ntargets, nfiles))
    rootfiles = [RootFile(filename) for filename in options.filenames]
    #### Create the output directory structure
    if not options.noclean and os.path.exists(options.output):
        shutil.rmtree(options.output)
    for path, folders, objects in walk_rootfile('', rootfiles[0], options):
        if not os.path.exists(joined(options.output, path)):
            os.makedirs(joined(options.output, path))
    #### Loop over plots to make, building the necessary calls
    plotargs = get_plot_inputs(rootfiles, options)
    call_lists = []
    ndigits = int(math.log10(len(plotargs))) + 1
    for i, (filenames, targets) in enumerate(plotargs):
        argstring = ', '.join(["'%s'" % x for x in (filenames + targets +
                                                    options.configs)])
        reduced_kwargs = dict(kwargs)
        for key, value in reduced_kwargs.items():
            if key in global_opts:
                del reduced_kwargs[key]
            elif type(value) is str:
                reduced_kwargs[key] = "'%s'" % value
        if 'numbering' in kwargs:
            reduced_kwargs['numbering'] = i + 1
        optstring = ', '.join(['%s=%s' % (key, value)
                               for key, value in reduced_kwargs.items()])
        if optstring:
            argstring = "%s, %s" % (argstring, optstring)
        plotpath, title, legentries = get_plotpath(filenames, targets)
        savepath = joined(options.output, plotpath)
        if 'numbering' in reduced_kwargs:
            dirs = savepath.split('/')
            dirs[-1] = str(i + 1).zfill(ndigits) + dirs[-1]
            savepath = '/'.join(dirs)
        call_vars = {'argstring' : argstring, 'ext' : options.ext,
                     'savepath' : savepath}
        if use_mpl:
            call_vars['trans'] = options.transparent
            call_vars['dpi'] = options.dpi
            api_call = ("figure, objects = "
                        "plotmpl(%(argstring)s)" % call_vars)
            save_call = ("figure.savefig('%(savepath)s', "
                         "transparent=%(trans)s, "
                         "dpi=%(dpi)s)" % call_vars)
        else:
            api_call = ("canvas, objects = "
                        "plot(%(argstring)s)" % call_vars)
            save_call = "canvas.SaveAs('%(savepath)s.%(ext)s')" % call_vars
        call_lists.append([api_call, save_call])
    #### Create scripts for that make the API calls
    ext = options.ext
    output = options.output
    processors = options.processors
    call_statements = '\n\n'.join([plotcall + '\n' + savecall 
                                   for plotcall, savecall in call_lists])
    allplots_script = allplots_template % locals()
    call_statements = "".join([multi_call_template % (plotcall, savecall) 
                               for plotcall, savecall in call_lists])
    allplots_multi_script = allplots_multi_template % locals()
    write_to_file(allplots_script, joined(options.output, 'allplots.py'))
    write_to_file(allplots_multi_script, 
                 joined(options.output, 'allplots_multi.py'))
    #### Execute the calls
    if use_multiprocessing:
        original_dir = os.getcwd()
        os.chdir(options.output)
        exec(allplots_multi_script)
        os.chdir(original_dir)
    else:
        for i, call_list in enumerate(call_lists):
            make_calls(*call_list)
            report_progress(i + 1, len(plotargs), options.output, options.ext)
        report_progress(len(plotargs), len(plotargs),
                        options.output, options.ext)
        print ''
    ## clean out empty directories
    for root, dirs, files in os.walk(options.output):
        if not os.listdir(root):
            os.rmdir(root)
    ## add index.html files to all directories
    if options.ext in ['png', 'gif', 'svg']:
        print "Writing html index files..."
        width, height = options.size
        if use_mpl:
            width, height = [x * options.dpi for x in options.size]
        for path, dirs, files in os.walk(options.output):
            dirs, files = sorted(dirs), sorted(files)
            make_html_index(path, dirs, files, options.ext,
                            options.html_template, options.ncolumns_html,
                            width, height)
    if options.merge:
        merge_pdf(options)


##############################################################################
######## Implementation ######################################################

def write_to_file(script, destination):
    f = open(destination, 'w')
    f.write(script)
    f.close()

def make_calls(api_call, save_call):
    exec(api_call)
    exec(save_call)

def option_diff(default, modified):
    #### Return a dict with the values from modified not present in default.
    diff = {}
    for key in dir(default):
        default_val = getattr(default, key)
        modified_val = getattr(modified, key)
        if (type(default_val) in [int, float, str, bool, type(None)] and
            key in dir(modified) and default_val != modified_val):
            diff[key] = modified_val
    return diff

def config_string():
    s = config_template
    if use_mpl:
        s = re.sub('root::.*\n', '', s)
        s = s.replace('mpl:::', '')
        s = s.replace('%prog', 'rootplotmpl')
    else:
        s = re.sub('mpl:::.*\n', '', s)
        s = s.replace('root::', '')
        s = s.replace('%prog', 'rootplot')
    return s

def write_config():
    if use_mpl:
        filename = 'rootplotmpl_config.py'
    else:
        filename = 'rootplot_config.py'
    f = open(filename, 'w')
    f.write(config_string())
    f.close()
    print "Wrote %s to the current directory" % filename
    sys.exit(0)

def add_from_config_files(options, configs):
    def append_to_options(config, options):
        for attribute in dir(config):
            if '__' not in attribute:
                attr = getattr(config, attribute)
                setattr(options, attribute, attr)
    configdir = tempfile.mkdtemp()
    sys.path.insert(0, '')
    sys.path.insert(0, configdir)
    f = open(joined(configdir, 'default_config.py'), 'w')
    f.write(config_string())
    f.close()
    import default_config
    append_to_options(default_config, options)
    if use_mpl:
        rc_name = 'rootplotmplrc'
    else:
        rc_name = 'rootplotrc'
    rc_path = os.path.expanduser('~/.%s' % rc_name)
    if os.path.exists(rc_path):
        print "Using styles and options from ~/.%s" % rc_name
        shutil.copy(rc_path, joined(configdir, '%s.py' % rc_name))
        configs.insert(0, '%s.py' % rc_name)
    for f in configs:
        user_config = __import__(f[:-3])
        append_to_options(user_config, options)
    shutil.rmtree(configdir)

def walk_rootfile(path, rootfile, options):
    #### Yield (path, folders, objects) for each directory under path.
    keys = rootfile.file.GetDirectory(path).GetListOfKeys()
    folders, objects = [], []
    for key in keys:
        name = key.GetName()
        classname = key.GetClassName()
        newpath = joined(path, name)
        dimension = 0
        matches_path = re.match(options.path, newpath)
        if 'TDirectory' in classname:
            folders.append(name)
        elif ('TH1' in classname or 'TGraph' in classname or
            classname == 'TProfile'):
            dimension = 1
        elif options.draw2D and ('TH2' in classname or 
                                 classname == 'TProfile2D'):
            dimension = 2
        if (matches_path and dimension):
            objects.append(name)
    yield path, folders, objects
    for folder in folders:
        for x in walk_rootfile(joined(path, folder), rootfile, options):
            yield x
        
def get_plot_inputs(files, options):
    #### Return a list of argument sets to be sent to plot().
    target_lists = []
    if options.targets:
        for path, folders, objects in walk_rootfile('', files[0], options):
            if path == options.targets[0]:
                #### targets are folders
                for obj in objects:
                    target_lists.append([joined(t, obj) 
                                         for t in options.targets])
    else:
        target_sets = [set() for f in files]
        for i, f in enumerate(files):
            for path, folders, objects in walk_rootfile('', f, options):
                for obj in objects:
                    target_sets[i].add(joined(path, obj))
        target_set = target_sets[0]
        for s in target_sets:
            target_set &= s
        target_lists = [[t] for t in target_set]
    if not target_lists:
        return [(options.filenames, options.targets)]
    else:
        return [(options.filenames, list(t)) for t in target_lists]

def fill_options(args, kwargs, scope):
    options = parse_arguments(args, scope=scope)
    for key, value in kwargs.items():
        options[key] = value
    options.process_configs(scope=scope)
    options.size = parse_size(options.size)
    options.split = options.ratio_split or options.efficiency_split
    options.ratio = (options.ratio or options.efficiency or
                     options.ratio_split or options.efficiency_split)
    options.efficiency = options.efficiency or options.efficiency_split
    if len(options.filenames) > 1 or len(options.targets) > 1:
        options.draw2D = None
    return options

def plot_hists_root(hists, options):
    #### Create a plot.
    canvas = ROOT.TCanvas("canvas", "", 
                          int(options.size[0]), int(options.size[1]))
    isTGraph = 'TGraph' in hists[0].rootclass
    objects = {'pads': [canvas]}
    if options.ratio:
        if options.split:
            objects['pads'] = divide_canvas(canvas, options.ratio_fraction)
            objects['pads'][0].cd()
        else:
            hists = make_ratio_hists(hists, options, options.ratio - 1)
            isTGraph = True
    if options.xerr:
        ROOT.gStyle.SetErrorX()
    histmax, first_draw, roothists = None, True, []
    if isTGraph:
        objects['multigraph'] = ROOT.TMultiGraph()
    else:
        objects['stack'] = ROOT.THStack(
            'st%s' % os.path.basename(options.plotpath),
            '%s;%s;%s' % (hists[0].title,
                          hists[0].xlabel, hists[0].ylabel))
    for i, hist in enumerate(hists):
        if not hist: continue
        name = "%s_%i" % (options.plotpath, i)
        if isTGraph:
            roothist = hist.TGraph(name=name)
        elif type(hist) is Hist:
            roothist = hist.TH1F(name=name.replace('/', '__'))
        else:
            roothist = hist.TH2F(name=name)
        roothist.SetLineStyle(options.line_styles[i])
        roothist.SetLineColor(options.line_colors[i])
        roothist.SetFillColor(options.fill_colors[i])
        roothist.SetMarkerColor(options.marker_colors[i])
        roothist.SetFillStyle(options.fill_styles[i])
        roothist.SetMarkerStyle(options.marker_styles[i])
        roothist.SetMarkerSize(options.marker_sizes[i])
        roothists.append(roothist)
        if (type(hist) is Hist and not isTGraph and 
            'stack' in options.draw_commands[i]):
            objects['stack'].Add(roothist)
    if 'stack' in objects and objects['stack'].GetHists():
        histmax = objects['stack'].GetMaximum()
    for roothist in roothists:
        histmax = max(histmax, roothist.GetMaximum())
    dimension = 1
    if type(hist) == Hist2D:
        dimension = 2
    if options.gridx or options.grid:
        for pad in objects['pads']:
            pad.SetGridx(not pad.GetGridx())
    if options.gridy or options.grid:
        objects['pads'][0].SetGridy(not objects['pads'][0].GetGridy())
    objects['legend'] = ROOT.TLegend(*parse_legend_root(options))
    for com in options.draw_commands:
        if 'stack' in com:
            first_draw = prep_first_draw(objects['stack'], histmax, options)
            com = com.replace('stack', '')
            objects['stack'].Draw(com)
            break
    for i, roothist in enumerate(roothists):
        if isTGraph:
            objects['multigraph'].Add(roothist)
        elif dimension == 1:
            if 'stack' not in options.draw_commands[i]:
                if first_draw:
                    first_draw = prep_first_draw(roothist, histmax, options)
                    roothist.Draw(options.draw_commands[i])
                else:
                    roothist.Draw("same " + options.draw_commands[i])
        else:
            roothist.Draw(options.draw2D)
        legendopt = 'lp'
        if options.fill_styles[i]: legendopt += 'f'
        if 'e' in options.draw_commands[i]: legendopt += 'e'
        objects['legend'].AddEntry(roothist, options.legend_entries[i],
                                   legendopt)
    if isTGraph:
        objects['multigraph'].Draw(options.draw_graph)
        prep_first_draw(objects['multigraph'], histmax, options)
        objects['multigraph'].Draw(options.draw_graph)
    if options.split and dimension == 1:
        objects['pads'][1].cd()
        objects['ratio_multigraph'] = plot_ratio_root(
            hists, roothist.GetXaxis().GetTitle(), options)
        xmin = hists[0].xedges[0]
        xmax = hists[0].xedges[-1]
        objects['ratio_multigraph'].GetXaxis().SetRangeUser(xmin, xmax)
        objects['pads'][0].cd()
    if options.logx:
        for pad in objects['pads']:
            pad.SetLogx(True)
    if options.logy:
        objects['pads'][0].SetLogy(True)
    if options.ratio_logy:
        if len(objects['pads']) > 1:
            objects['pads'][1].SetLogy(True)
    if options.numbering:
        display_page_number(options)
    if roothist.InheritsFrom('TH1'):
        if options.overflow:
            display_overflow(objects['stack'], roothist)
        if options.underflow:
            display_underflow(objects['stack'], roothist)
    if options.legend_location and dimension == 1:
        objects['legend'].Draw()
    exec(options.decoration_root)
    objects['hists'] = roothists
    return canvas, objects

def plot_hists_mpl(hists, options):
    #### Create a plot.
    fig = plt.figure(1, figsize=options.size)
    fig.clf()     # clear figure
    axes = plt.axes()
    objects = {'axes' : [axes, axes]}
    if options.ratio:
        if options.split:
            objects['axes'] = divide_axes(fig, axes, options.ratio_fraction)
            axes = objects['axes'][0]
            fig.sca(axes)
        else:
            hists = make_ratio_hists(hists, options, options.ratio - 1)
    refhist = hists[0]
    if refhist is None:
        refhist = hists[1]
    fullstack, objects['stack'] = HistStack(), HistStack()
    histmax, allempty = None, True
    for i, hist in enumerate(hists):
        if hist and hist.entries:
            allempty = False
        if type(hist) is Hist:
            # Avoid errors due to zero bins with log y axis
            if options.logy and options.plot_styles[i] != 'errorbar':
                for j in range(hist.nbins):
                    hist.y[j] = max(hist.y[j], 1e-10)
            if options.plot_styles[i] in ['barh', 'barcluster', 'stack']:
                objects['stack'].add(hist, log=options.logy,
                                     hatch=options.fill_styles[i],
                                     linestyle=options.line_styles[i],
                                     edgecolor=options.line_colors[i],
                                     facecolor=options.fill_colors[i])
            fullstack.add(hist)
    if 'stack' in options.plot_styles:
        histmax = max(histmax, objects['stack'].stackmax())
    elif 'barh' in options.plot_styles or 'barcluster' in options.plot_styles:
        histmax = max(histmax, objects['stack'].max())
    for hist in fullstack:
        histmax = max(histmax, max(hist))
    if allempty:
        fig.text(0.5, 0.5, "No Entries", ha='center', va='center')
    elif type(refhist) is Hist:
        for i, hist in enumerate(hists):
            if hist:
                if options.plot_styles[i] == "errorbar":
                    if options.logy:
                        axes.set_yscale('log')
                        # Logy would fail if hist all zeroes
                        if not np.nonzero(hist.y)[0].tolist():
                            continue
                        # Errorbars get messed up when they extend down to zero
                        for j in range(hist.nbins):
                            yerr = hist.yerr[0][j]
                            if (hist[j] - yerr) < (0.01 * yerr):
                                hist.yerr[0][j] *= 0.99
                    hist.errorbar(fmt=options.marker_styles[i],
                                  yerr=True,
                                  xerr=options.xerr,
                                  markersize=options.marker_sizes[i],
                                  color=options.fill_colors[i],
                                  ecolor=options.errorbar_colors[i],
                                  label_rotation=options.xlabel_rotation,
                                  label_alignment=options.xlabel_alignment)
                elif options.plot_styles[i] == "bar":
                    hist.bar(alpha=options.alphas[i], 
                             log=options.logy,
                             width=options.barwidth,
                             hatch=options.fill_styles[i],
                             edgecolor=options.line_colors[i],
                             facecolor=options.fill_colors[i],
                             label_rotation=options.xlabel_rotation,
                             label_alignment=options.xlabel_alignment)
                elif 'hist' in options.plot_styles[i]:
                    histtype = 'step'
                    if 'fill' in options.plot_styles[i]:
                        histtype = 'stepfilled'
                    hist.hist(alpha=options.alphas[i],
                              histtype=histtype,
                              log=options.logy,
                              hatch=options.fill_styles[i],
                              edgecolor=options.line_colors[i],
                              facecolor=options.fill_colors[i],
                              label_rotation=options.xlabel_rotation,
                              label_alignment=options.xlabel_alignment)
                if options.logx:
                    for ax in objects['axes']:
                        ax.set_xscale('log')
        if objects['stack'].hists:
            if 'stack' in options.plot_styles:
                objects['stack'].histstack(
                    histtype='stepfilled',
                    label_rotation=options.xlabel_rotation,
                    label_alignment=options.xlabel_alignment)
            elif 'barh' in options.plot_styles:
                objects['stack'].barh(
                    width=options.barwidth,
                    label_rotation=options.xlabel_rotation,
                    label_alignment=options.xlabel_alignmenth)
            elif 'barcluster' in options.plot_styles:
                objects['stack'].barcluster(
                    width=options.barwidth,
                    label_rotation=options.xlabel_rotation,
                    label_alignment=options.xlabel_alignment)
        if 'barh' not in options.plot_styles:
            axes.set_xlim(refhist.xedges[0], refhist.xedges[-1])
        if options.logy:
            my_min = fullstack.min(threshold=1.1e-10)
            rounded_min = 1e100
            while (rounded_min > my_min):
                rounded_min /= 10
            axes.set_ylim(ymin=rounded_min)
        if options.xmin is not None:
            axes.set_xlim(xmin=options.xmin)
        if options.xmax is not None:
            axes.set_xlim(xmax=options.xmax)
        if options.ymin is not None:
            axes.set_ylim(ymin=options.ymin)
        if options.ymax is not None:
            axes.set_ylim(ymax=options.ymax)
        elif ('barh' not in options.plot_styles and 
              histmax != 0 and not options.ymax):
            axes.set_ylim(ymax=histmax * options.top_padding_factor)
        if options.overflow:
            axes.text(hist.x[-1], axes.set_ylim()[0], options.overflow_text,
                      rotation='vertical', ha='center',
                      alpha=options.overflow_alpha, size=options.overflow_size)
        if options.underflow:
            axes.text(hist.x[0], axes.set_ylim()[0], options.underflow_text,
                      rotation='vertical', ha='center',
                      alpha=options.overflow_alpha, size=options.overflow_size)
        if options.gridx or options.grid:
            axes.xaxis.grid()
        if options.gridy or options.grid:
            axes.yaxis.grid()
        if (options.legend_location != 'None' or options.legend_axes_bbox or 
            options.legend_figure_bbox):
            try:
                options.legend_location = int(options.legend_location)
            except ValueError:
                pass
            if options.legend_axes_bbox:
                kwargs = {'bbox_to_anchor' : options.legend_axes_bbox}
            elif options.legend_figure_bbox:
                kwargs = {'bbox_to_anchor' : options.legend_figure_bbox,
                          'bbox_transform' : fig.transFigure}
            else:
                kwargs = {'loc' : options.legend_location}
            if options.legend_ncols:
                kwargs['ncol'] = int(options.legend_ncols)
            objects['legend'] = axes.legend(numpoints=1, **kwargs)
    elif type(refhist) is Hist2D:
        drawfunc = getattr(hist, options.draw2D)
        if 'col' in options.draw2D:
            if options.cmap:
                drawfunc(cmap=options.cmap)
            else:
                drawfunc()
        else:
            drawfunc(color=options.fill_colors[0])
    axes.set_title(r2m.replace(refhist.title, options.replace))
    if 'barh' in options.plot_styles:
        set_ylabel = objects['axes'][1].set_xlabel
        set_xlabel = axes.set_ylabel
    else:
        set_ylabel = axes.set_ylabel
        set_xlabel = objects['axes'][1].set_xlabel
    if options.ratio and not options.split and not options.ylabel:
        ratio_file = options.legend_entries[options.ratio - 1]
        if options.efficiency:
            set_ylabel(options.efficiency_label % locals())
        else:
            set_ylabel(options.ratio_label % locals())
    else:
        set_ylabel(r2m.replace(refhist.ylabel, options.replace))
    set_xlabel(r2m.replace(refhist.xlabel, options.replace))
    if options.split:
        fig.sca(objects['axes'][1])
        plot_ratio_mpl(objects['axes'][1], hists, options)
        fig.sca(objects['axes'][0])
    if options.numbering:
        fig.text(options.numbering_x_mpl, options.numbering_y_mpl,
                 options.numbering, size=options.numbering_size_mpl,
                 ha=options.numbering_ha_mpl, va=options.numbering_va_mpl)
    options.decoration_mpl(fig, objects['axes'], options.plotpath,
                           options, hists)
    objects['hists'] = hists
    return fig, objects

def initialize_hists(args, kwargs):
    options = fill_options(args, kwargs, scope='plot')
    if use_mpl:
        load_matplotlib(options.ext)
    nfiles = len(options.filenames)
    ntargets = len(options.targets)
    if nfiles < 1:
        raise TypeError("plot() takes at least 1 filename argument (0 given)")
    elif ntargets > 1 and nfiles > 1:
        raise TypeError("plot() takes exactly 1 target (%i given) when "
                        "multiple files are specified (%i given)" %
                        (nfiles, ntargets))
    options.nhists = max(nfiles, ntargets)
    process_options(options)
    files = [RootFile(filename) for filename in options.filenames]
    hists = []
    stack_integral = 0.
    for i, (f, target) in enumerate(cartesian_product(files, options.targets)):
        try:
            roothist = f.file.Get(target)
        except ReferenceError:
            hists.append(None)
            continue
        try:
            isTGraph = not roothist.InheritsFrom('TH1')
        except TypeError:
            raise TypeError("'%s' does not appear to be a valid target" %
                            target)
        dimension = 1
        if use_mpl:
            stacked = 'stack' in options.plot_styles[i]
        else:
            stacked = 'stack' in options.draw_commands[i]
        if not isTGraph:
            dimension = roothist.GetDimension()
            roothist.Scale(options.scale[i])
            if options.rebin:
                roothist.Rebin(options.rebin)
        title, xlabel, ylabel = get_labels(roothist, options)
        if options.normalize:
            norm_file = options.legend_entries[options.normalize - 1]
            ylabel = options.target_normalized_title % locals()
        if options.area_normalize:
            ylabel = options.area_normalized_title
        if dimension == 1:
            hist = Hist(roothist, label=options.legend_entries[i],
                        title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            hist = Hist2D(roothist, label=options.legend_entries[i],
                          title=title, xlabel=xlabel, ylabel=ylabel)
        if stacked:
            stack_integral += roothist.Integral()
        roothist.Delete()
        hists.append(hist)
    for i, hist in enumerate(hists):
        if use_mpl:
            stacked = 'stack' in options.plot_styles[i]
        else:
            stacked = 'stack' in options.draw_commands[i]
        if dimension == 1:
            if options.overflow:
                hist.y[-1] += hist.overflow
            if options.underflow:
                hist.y[0] += hist.underflow
            if options.area_normalize:
                if sum(hist.y):
                    if stacked:
                        hist.scale(1./stack_integral)
                    else:
                        hist.scale(1./sum(hist.y))
            if options.normalize:
                numerhist = hists[options.normalize - 1]
                if options.norm_range:
                    lowbin, highbin = parse_range(hist.xedges,
                                                  options.norm_range)
                    numer = numerhist.TH1F().Integral(lowbin, highbin)
                    denom = hist.TH1F().Integral(lowbin, highbin)
                else:
                    numer = sum(numerhist.y)
                    denom = sum(hist.y)
                    if stacked:
                        denom = stack_integral
                if denom:
                    hist.scale(numer / denom)
    return hists, options

def parse_size(size_option):
    #### Return a width and height parsed from size_option.
    try:
        xpos = size_option.find('x')
        return float(size_option[:xpos]), float(size_option[xpos+1:])
    except TypeError:
        return size_option

def parse_color(color, tcolor=False):
    #### Return an rgb tuple or a ROOT TColor from a ROOT color index or
    #### an rgb(a) tuple.
    if color is None:
        return None
    elif color == 'none' or color == 'None':
        return 'none'
    r, g, b = 0, 0, 0
    try:
        color = ROOT.gROOT.GetColor(color)
        r, g, b = color.GetRed(), color.GetGreen(), color.GetBlue()
    except TypeError:
        try:
            if max(color) > 1.:
                color = [x/256. for x in color][0:3]
        except TypeError:
            pass
        try:
            color = mpal.colors.colorConverter.to_rgb(color)
        except NameError:
            pass
        r, g, b = color[0:3]
    if tcolor:
        return ROOT.TColor.GetColor(r, g, b)
    return (r, g, b)

def get_labels(hist, options):
    #### Return the appropriate histogram and axis titles for hist.
    title = hist.GetTitle().split(';')[0]
    xlabel = hist.GetXaxis().GetTitle()
    ylabel = hist.GetYaxis().GetTitle()
    if options.title:
        if options.title.startswith('+'):
            title += options.title[1:]
        else:
            title = options.title
    if options.xlabel:
        if options.xlabel.startswith('+'):
            xlabel += options.xlabel[1:]
        else:
            xlabel = options.xlabel
    if options.ylabel:
        if options.ylabel.startswith('+'):
            ylabel += options.ylabel[1:]
        else:
            ylabel = options.ylabel
    return title, xlabel, ylabel

def report_progress(counter, nplots, output, ext, divisor=1):
    #### Print the current number of finished plots.
    if counter % divisor == 0:
        print("\r%i plots of %i written to %s/ in %s format" %
              (counter, nplots, output, ext)),
        sys.stdout.flush()

def merge_pdf(options):
    #### Merge together all the produced plots into one pdf file.
    destination = joined(options.output, 'allplots.pdf')
    paths = []
    for path, dirs, files in os.walk(options.output):
        paths += [joined(path, x) for x in files if x.endswith('.pdf')]
    if not paths:
        print "No output files, so no merged pdf was made"
        return
    print "Writing %s..." % destination
    os.system('gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite '
              '-dAutoRotatePages=/All '
              '-sOutputFile=%s %s' % (destination, ' '.join(paths)))

def display_page_number(options):
    #### Add a page number to the top corner of the canvas.
    page_text = ROOT.TText()
    page_text.SetTextSize(options.numbering_size_root)
    page_text.SetTextAlign(options.numbering_align_root)
    page_text.DrawTextNDC(options.numbering_x_root, options.numbering_y_root,
                          '%i' % options.numbering)

def display_overflow(stack, hist):
    #### Add the overflow to the last bin and print 'Overflow' on the bin.
    nbins = hist.GetNbinsX()
    x = 0.5 * (hist.GetBinLowEdge(nbins) +
               hist.GetBinLowEdge(nbins + 1))
    y = stack.GetMinimum('nostack')
    display_bin_text(x, y, nbins, 'Overflow')

def display_underflow(stack, hist):
    #### Add the underflow to the first bin and print 'Underflow' on the bin.
    nbins = hist.GetNbinsX()
    x = 0.5 * (hist.GetBinLowEdge(1) +
               hist.GetBinLowEdge(2))
    y = stack.GetMinimum('nostack')
    display_bin_text(x, y, nbins, 'Underflow')

def display_bin_text(x, y, nbins, text):
    #### Overlay TEXT on this bin.
    bin_text = ROOT.TText()
    bin_text.SetTextSize(min(1. / nbins, 0.04))
    bin_text.SetTextAlign(12)
    bin_text.SetTextAngle(90)
    bin_text.SetTextColor(13)
    bin_text.SetTextFont(42)
    bin_text.DrawText(x, y, text)

def prep_first_draw(hist, histmax, options):
    #### Set all the pad attributes that depend on the first object drawn.
    hist.SetMaximum(histmax * options.top_padding_factor)
    if options.xmin is not None and options.xmax is not None:
        hist.GetXaxis().SetRangeUser(options.xmin, options.xmax)
    elif options.xmin is not None:
        original_max = hist.GetBinLowEdge(hist.GetNbinsX() + 1)
        hist.GetXaxis().SetRangeUser(options.xmin, original_max)
    elif options.xmax is not None:
        original_min = hist.GetBinLowEdge(1)
        hist.GetXaxis().SetRangeUser(original_min, options.xmax)
    if options.ymin is not None:
        hist.SetMinimum(options.ymin)
    if options.ymax is not None:
        hist.SetMaximum(options.ymax)
    if options.ratio:
        if options.split:
            hist.Draw()
            hist.GetXaxis().SetBinLabel(1, '') # Don't show tick labels
            if ';' in hist.GetTitle():
                # dealing with bug in THStack title handling
                titles = hist.GetTitle().split(';')
                if len(titles) > 1: titles[1] = ''
                hist.SetTitle(';'.join(titles))
            else:
                hist.GetXaxis().SetTitle('')
            ## Avoid overlap of y-axis numbers by supressing zero
            if (not options.logy and
                hist.GetMaximum() > 0 and
                hist.GetMinimum() / hist.GetMaximum() < 0.25):
                hist.SetMinimum(hist.GetMaximum() / 10000)
        else:
            ratio_file = options.legend_entries[options.ratio - 1]
            if options.efficiency:
                hist.GetYaxis().SetTitle(options.efficiency_label % locals())
            else:
                hist.GetYaxis().SetTitle(options.ratio_label % locals())
    return False

def divide_canvas(canvas, ratio_fraction):
    #### Divide the canvas into two pads; the bottom is ratio_fraction tall.
    ## Both pads are set to the full canvas size to maintain font sizes
    ## Fill style 4000 used to ensure pad transparency because of this
    margins = [ROOT.gStyle.GetPadTopMargin(), ROOT.gStyle.GetPadBottomMargin()]
    useable_height = 1 - (margins[0] + margins[1])
    canvas.Clear()
    pad = ROOT.TPad('mainPad', 'mainPad', 0., 0., 1., 1.)
    pad.SetFillStyle(4000)
    pad.Draw()
    pad.SetBottomMargin(margins[1] + ratio_fraction * useable_height)
    pad_ratio = ROOT.TPad('ratioPad', 'ratioPad', 0., 0., 1., 1.);
    pad_ratio.SetFillStyle(4000)
    pad_ratio.Draw()
    pad_ratio.SetTopMargin(margins[0] + (1 - ratio_fraction) * useable_height)
    return pad, pad_ratio

def divide_axes(fig, axes, ratio_fraction):
    #### Create two subaxes, the lower one taking up ratio_fraction of total.
    x1, y1, x2, y2 = axes.get_position().get_points().flatten().tolist()
    width = x2 - x1
    height = y2 - y1
    lower_height = height * ratio_fraction
    upper_height = height - lower_height
    lower_axes = fig.add_axes([x1, y1, width, lower_height], axisbg='None')
    upper_axes = fig.add_axes([x1, y1 + lower_height, width, upper_height],
                              axisbg='None', sharex=lower_axes)
    ## Make original axes and the upper ticklabels invisible
    axes.set_xticks([])
    axes.set_yticks([])
    plt.setp(upper_axes.get_xticklabels(), visible=False)
    return upper_axes, lower_axes

def make_ratio_hists(hists, options, ratio_index):
    denom = hists[ratio_index]
    if options.efficiency:
        ratios = [hist.divide_wilson(denom) for hist in hists]
    else:
        ratios = [hist.divide(denom) for hist in hists]        
    ratios[ratio_index] = None
    return ratios

def plot_ratio_root(hists, xlabel, options):
    #### Plot the ratio of each hist in hists to the ratio_indexth hist.
    ratio_index = options.ratio - 1
    ratio_file = options.legend_entries[ratio_index]
    if options.efficiency:
        ylabel = options.efficiency_label % locals()
    else:
        ylabel = options.ratio_label % locals()
    multigraph = ROOT.TMultiGraph("ratio_multi",
                                  ";%s;%s" % (xlabel, ylabel))
    if options.stack and options.data:
        numerator = hists[ratio_index]
        hists = hists[:]
        hists.pop(ratio_index)
        denominator = hists[0]
        for hist in hists[1:]:
            denominator += hist
        hists = [numerator, denominator]
        ratio_index = 1
    for i, ratio_hist in enumerate(make_ratio_hists(hists, options, 
                                                    ratio_index)):
        if i == ratio_index:
            continue
        graph = ratio_hist.TGraph()
        graph.SetLineColor(options.line_colors[i])
        graph.SetMarkerColor(options.marker_colors[i])
        graph.SetMarkerStyle(options.marker_styles[i])
        graph.SetMarkerSize(options.marker_sizes[i])
        multigraph.Add(graph)
    multigraph.Draw(options.draw_graph)
    multigraph.GetYaxis().SetNdivisions(507) # Avoids crowded labels
    if options.ratio_max is not None: multigraph.SetMaximum(options.ratio_max)
    if options.ratio_min is not None: multigraph.SetMinimum(options.ratio_min)
    multigraph.Draw(options.draw_graph)
    return multigraph

def plot_ratio_mpl(axes, hists, options):
    #### Plot the ratio of each hist in hists to the ratio_indexth hist.
    ratio_index = options.ratio - 1
    stack = HistStack()
    if options.stack and options.data:
        numerator = hists[ratio_index]
        hists = hists[:]
        hists.pop(ratio_index)
        denominator = hists[0]
        for hist in hists[1:]:
            denominator += hist
        hists = [numerator, denominator]
        ratio_index = 1
    for i, ratio_hist in enumerate(make_ratio_hists(hists, options, 
                                                    ratio_index)):
        if i == ratio_index:
            continue
        ratio_hist.y = [item or 0 for item in ratio_hist.y] ## Avoid gaps
        stack.add(ratio_hist, fmt=options.marker_styles[i],
                  color=options.fill_colors[i],
                  ecolor=options.errorbar_colors[i])
    if options.ratio_logy:
        axes.set_yscale('log')
    stack.errorbar(yerr=True)
    axes.yaxis.set_major_locator(
        mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    if options.ratio_max is not None: axes.set_ylim(ymax=options.ratio_max)
    if options.ratio_min is not None: axes.set_ylim(ymin=options.ratio_min)
    ratio_file = options.legend_entries[ratio_index]
    if options.efficiency:
        axes.set_ylabel(options.efficiency_label % locals())
    else:
        axes.set_ylabel(options.ratio_label % locals())
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position('right')
    axes.yaxis.label.set_rotation(-90)

def make_html_index(path, dirs, files, filetype, template, ncolumns, 
                    width, height):
    files = [x for x in files if x.endswith(filetype)]
    output_file = open(joined(path, 'index.html'), 'w')
    previous_dirs = [x for x in path.split('/') if x]
    ndirs = len(previous_dirs)
    back_nav = ['<a href="%s">%s</a>' %
                ('../' * (ndirs - i - 1) + 'index.html', previous_dirs[i])
                for i in range(ndirs)]
    back_nav = '/'.join(back_nav) + '/'
    forward_nav = ['<li><a href="%s/index.html">%s</a>/</li>' % (x, x)
                   for x in dirs]
    forward_nav = '\n    '.join(forward_nav)
    imgtemplate = '<a name="%(plot)s"><a href="index.html#%(plot)s">'
    if filetype.lower() == 'svg':
        imgtemplate += ('<object type="image/svg+xml" data="%(plot)s" '
                        'width=%(width)i height=%(height)i></object>')
    else:
        imgtemplate += '<img src="%(plot)s" height=%(height)i width=%(width)i>'
    imgtemplate += '</a></a>'
    plots = '\n'
    for plot in files:
        plots += imgtemplate % locals() + '\n'
    plots = re.sub('((\\n<a.*){%i})' % ncolumns, r'\1<br>', plots)
    output_file.write(template % locals())
    output_file.close()
    
def parse_range(xedges, expression):
    #### Returns the indices of the low and high bins indicated in expression.
    closest = lambda l,y: l.index(min(l, key=lambda x:abs(x-y)))
    match = re.match(r'([^x]*)x([^x]*)', expression)
    lower, upper = float(match.group(1)), float(match.group(2))
    lowbin = closest(xedges, lower) + 1
    highbin = closest(xedges, upper)
    return lowbin, highbin

def parse_legend_root(options):
    #### Return the corners to use for the legend based on options.
    legend_height = min(options.legend_entry_height * options.nhists + 0.02,
                        options.max_legend_height)
    if type(options.legend_location) is int:
        options.legend_location = options.legend_codes[options.legend_location]
    elif options.legend_location.lower() == 'none':
        options.legend_location = None
    if options.legend_location:
        if 'upper' in options.legend_location:
            top = options.legend_upper_bound
            bottom = options.legend_upper_bound - legend_height
        elif 'lower' in options.legend_location:
            bottom = options.legend_lower_bound
            top = options.legend_lower_bound + legend_height
        else:
            top = 0.5 + legend_height / 2
            bottom = 0.5 - legend_height / 2
        if 'left' in options.legend_location:
            left = options.legend_left_bound
            right = options.legend_left_bound + options.legend_width
        elif 'right' in options.legend_location:
            right = options.legend_right_bound
            left = options.legend_right_bound - options.legend_width
        else:
            right = 0.5 + options.legend_width / 2
            left = 0.5 - options.legend_width / 2
        return [left, bottom, right, top]
    return [0, 0, 0, 0]

def load_matplotlib(ext):
    if 'mpl' not in globals().keys():
        global r2m, mpl, np, plt
        try:
            import matplotlib as mpl
        except ImportError:
            print "Unable to access matplotlib"
            sys.exit(1)
        import numpy as np
        mpldict = {'png' : 'AGG',
                   'pdf' : 'PDF',
                   'ps'  : 'PS',
                   'svg' : 'SVG'}
        if ext not in mpldict:
            raise ValueError("%s is not an output type recognized by "
                             "matplotlib" % ext)
        mpl.use(mpldict[ext])
        global Hist, Hist2D, HistStack
        import rootplot.root2matplotlib as r2m
        from rootplot.root2matplotlib import Hist, Hist2D, HistStack
        import matplotlib.pyplot as plt

def samebase(targets):
    for target in targets:
        if os.path.basename(target) != os.path.basename(targets[0]):
            return False
    return True

def allsame(targets):
    for target in targets:
        if target != targets[0]:
            return False
    return True

def diffpart(targets):
    targets = [target.split('/') for target in targets]
    for i in range(len(targets[0])):
        for target in targets:
            if target[i] != targets[0][i]:
                return ['/'.join(target[i:]) for target in targets]

def get_plotpath(filenames, targets):
    if len(targets) >= 2:
        diffs = diffpart(targets)
        if not allsame([d.split('/')[-1] for d in diffs]):
            plotpath = 'plot'
            title = 'plot'
            legentries = diffs
        else:
            plotpath = '/'.join(diffs[0].split('/')[1:])
            title = diffs[0].split('/')[-1]
            legentries = [d.split('/')[0] for d in diffs]
    else:
        plotpath = targets[0]
        title = ''
        legentries = [f[:-5] for f in filenames]
    return plotpath, title, legentries

def process_options(options):
    #### Refine options for this specific plot, based on plotname
    def comma_separator(obj, objtype, nhists):
        #### Split a comma-separated string into a list.
        if type(obj) is list:
            return obj
        if type(obj) is str and ',' in obj:
            try:
                return [objtype(x) for x in obj.split(',')]
            except TypeError:
                return [eval(objtype)(x) for x in obj.split(',')]
        try:
            return [objtype(obj) for i in range(nhists)]
        except TypeError:
            return [eval(objtype)(obj) for i in range(nhists)]
    nhists = options.nhists
    if options.targets:
        plotpath, title, legentries = get_plotpath(options.filenames,
                                                   options.targets)
        options.plotpath = plotpath
        if not options.title:
            options.title = title
        if not options.legend_entries:
            options.legend_entries = legentries
        options.legend_entries = comma_separator(options.legend_entries,
                                                 str, nhists)
    options.scale = comma_separator(options.scale, float, nhists)
    if options.efficiency_split: options.ratio_max = 1.
    if nhists > 1: options.draw2D = None
    if use_mpl:
        plot_style = 'histfill'
        if options.bar: plot_style = 'bar'
        elif options.barh: plot_style = 'barh'
        elif options.barcluster: plot_style = 'barcluster'
        elif options.errorbar: plot_style = 'errorbar'
        elif options.hist: plot_style = 'hist'
        elif options.histfill: plot_style = 'histfill'
        if options.stack: plot_style = 'stack'
    if not options.markers and use_mpl:
        options.marker_styles = ['o' for i in xrange(nhists)]
    if not options.line_colors:
        options.line_colors = options.colors
    if not options.fill_colors:
        options.fill_colors = options.colors
    if not options.marker_colors:
        options.marker_colors = options.colors
    if use_mpl:
        if not options.line_styles:
            options.line_styles = ['solid' for i in xrange(nhists)]
        if not options.plot_styles:
            options.plot_styles = [plot_style for i in xrange(nhists)]
        if not options.errorbar_colors:
            options.errorbar_colors = [None for i in xrange(nhists)]
        if not options.alphas:
            options.alphas = [options.alpha for i in xrange(nhists)]
    else:
        if not options.line_styles:
            options.line_styles = [1 for i in xrange(nhists)]
        if not options.draw_commands:
            if options.stack:
                options.draw_commands = ['stack ' + options.draw
                                         for i in xrange(nhists)]
            else:
                options.draw_commands = [options.draw
                                         for i in xrange(nhists)]
    if not options.fill_styles:
        if use_mpl:
            options.fill_styles = [None for i in xrange(nhists)]
        else:
            if options.fill:
                options.fill_styles = [1001 for i in xrange(nhists)]
            else:
                options.fill_styles = [0 for i in xrange(nhists)]
    if not options.marker_sizes:
        if options.markers:
            if use_mpl: size = mpl.rcParams['lines.markersize']
            else: size = ROOT.gStyle.GetMarkerSize()
        else:
            size = 0
        options.marker_sizes = [size for i in xrange(nhists)]
    if options.data:
        i = options.data - 1
        options.line_styles[i] = options.data_linestyle
        options.line_colors[i] = options.data_color
        options.fill_colors[i] = options.data_color
        options.marker_colors[i] = options.data_color
        if use_mpl:
            options.plot_styles[i] = 'errorbar'
        else:
            options.fill_styles[i] = 0
            options.draw_commands[i] = 'e'
        options.marker_styles[i] = options.data_marker
        if not options.marker_sizes[i]:
            if use_mpl:
                options.marker_sizes[i] = mpl.rcParams['lines.markersize']
            else:
                options.marker_sizes[i] = ROOT.gStyle.GetMarkerSize()
        if nhists == 2 and options.mc_color:
            options.fill_colors[(i+1)%2] = options.mc_color
    for opt in [x for x in options.keys() if 'colors' in x]:
        try:
            colors = options[opt]
            options[opt] = [parse_color(x, not use_mpl) for x in colors]
        except AttributeError:
            pass
    if options.targets:
        #### Apply extra options based on hist name
        plotname = os.path.basename(options.plotpath)
        for option, value, regexes in options.options_by_histname:
            for regex in regexes:
                if re.match(regex, plotname):
                    setattr(options, option, value)
        #### Final setup
        if options.logy:
            if options.ymin <= 0:
                options.ymin = None
            options.top_padding_factor = options.top_padding_factor_log

def cartesian_product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def parse_arguments(argv, scope='global'):
    if use_mpl:
        import matplotlib as mpl
        figsize = 'x'.join([str(x) for x in mpl.rcParams['figure.figsize']])
    else:
        figsize = 'x'.join([str(ROOT.gStyle.GetCanvasDefW()),
                            str(ROOT.gStyle.GetCanvasDefH())])
    def opt(**kwargs):
        return kwargs
    def addopt(group, *args, **kwargs):
        if use_mpl and 'mpl' in kwargs:
            opts = kwargs['mpl']
            kwargs = dict(kwargs, **opts)
        if not use_mpl and 'root' in kwargs:
            opts = kwargs['root']
            kwargs = dict(kwargs, **opts)
        if 'opts' in locals():
            del kwargs['mpl']
            del kwargs['root']
        if 'metadefault' in kwargs:
            val = kwargs.pop('metadefault')
            kwargs['default'] = val
            kwargs['metavar'] = val
        if 'metavar' in kwargs and ' ' in str(kwargs['metavar']):
            kwargs['metavar']="'%s'" % kwargs['metavar']
        group.add_option(*args, **kwargs)

    help_formatter = optparse.IndentedHelpFormatter()
    parser = optparse.OptionParser(usage=usage, formatter=help_formatter,
                                   version='%s %s' % ('%prog', __version__))
    Group = optparse.OptionGroup
    g_control = Group(parser, "Control the overall behavior of rootplot")
    g_output  = Group(parser, "Control the output format")
    g_hist    = Group(parser, "Manipulate your histograms")
    g_style   = Group(parser, "Fine-tune your style")
    parser.add_option_group(g_control)
    parser.add_option_group(g_output)
    parser.add_option_group(g_hist)
    parser.add_option_group(g_style)
    #### Process control options
    addopt(g_control, '--config', action='store_true',
           help="do nothing but write a template configuration file "
           "called rootplot_config.py")
    addopt(g_control, '--debug', action='store_true',
           help="turn off error-catching to more easily identify errors")
    addopt(g_control, '--path', metavar="'.*'", default='.*',
           help="only process plot(s) matching this regular expression")
    addopt(g_control, '--processors', type='int',
           metadefault=find_num_processors(),
           help="the number of parallel plotting processes to create")
    #### Output options
    addopt(g_output, '-e', '--ext', metadefault='png',
           help="choose an output extension")
    addopt(g_output, '--merge', action='store_true', default=False,
           help="sets --ext=pdf and creates a merged file "
           "containing all plots")
    addopt(g_output, '--noclean', action='store_true', default=False,
           help="skips destroying the output directory before drawing")
    addopt(g_output, '--output', metadefault='plots', 
           help="name of output directory")
    addopt(g_output, '--numbering', action='store_true', default=False,
           help="print page numbers on images and prepend them to file names; "
           "numbering will respect the order of objects in the ROOT file")
    addopt(g_output, '--size', metadefault=figsize,
           root=opt(help="set the canvas size to 'width x height' in pixels"),
           mpl=opt(help="set the canvas size to 'width x height' in inches"))
    if use_mpl:
        addopt(g_output, '--dpi', type=float,
               metadefault=mpl.rcParams['figure.dpi'],
               help="set the resolution of the output files")
        addopt(g_output, '--transparent', action="store_true", default=False,
            help="use a transparent background")
    #### Histogram manipulation options
    addopt(g_hist, '-n', '--area-normalize', action='store_true',
           default=False, help="area-normalize the histograms")
    addopt(g_hist, '--scale', metavar='VAL', default=1.,
           help="normalize all histograms by VAL, or by individual values "
           "if VAL is a comma-separated list")
    addopt(g_hist, '--normalize', metavar='NUM', type='int', default=0,
           help="normalize to the NUMth target (starting with 1)")
    addopt(g_hist, '--norm-range', metavar='LOWxHIGH',
           help="only use the specified data range in determining "
           "the normalization")
    addopt(g_hist, '--rebin', metavar="N", type=int,
           help="group together bins in sets of N")
    addopt(g_hist, '--ratio', type='int', default=0, metavar='NUM',
           help="divide histograms by the NUMth target (starting from 1)")
    addopt(g_hist, '--ratio-split', type='int', default=0, metavar='NUM',
           help="same as --ratio, but split the figure in two, displaying "
           "the normal figure on top and the ratio on the bottom")
    addopt(g_hist, '--efficiency', type='int', default=0, metavar='NUM',
           help="same as --ratio, but with errors computed by the Wilson "
           "score interval")
    addopt(g_hist, '--efficiency-split', type='int', default=0, metavar='NUM',
           help="same as --ratio-split, but with errors computed by the Wilson "
           "score interval")
    #### Style options
    if not use_mpl:
        addopt(g_style, '--draw', metadefault='p H',
               help="argument to pass to ROOT's Draw command; try 'e' for "
               "errorbars, or 'hist' to make sure no errorbars appear")
    addopt(g_style, '--draw2D',
           root=opt(metadefault='box',
                    help="argument to pass to TH2::Draw (ignored if multiple "
                    "targets specified); set "
                    'to "" to turn off 2D drawing'),
           mpl=opt(metadefault='box', 
                   help="command to use for plotting 2D hists; (ignored if "
                   "multiple targets specified) "
                   "choose from 'contour', 'col', 'colz', and 'box'")
           )
    if not use_mpl:
        addopt(g_style, '-f', '--fill', action='store_true', default=False,
                          help="Histograms will have a color fill")
    if use_mpl:
        addopt(g_style, '--errorbar', action="store_true", default=False,
               help="output a matplotlib errorbar graph")
        addopt(g_style, '--barcluster', action="store_true", default=False,
               help="output a clustered bar graph")
        addopt(g_style, '--barh', action="store_true", default=False,
               help="output a horizontal clustered bar graph")
        addopt(g_style, '--bar', action="store_true", default=False,
               help="output a bar graph with all histograms overlaid")
        addopt(g_style, '--hist', action="store_true", default=False,
            help="output a matplotlib hist graph (no fill)")
        addopt(g_style, '--histfill', action="store_true", default=False,
            help="output a matplotlib hist graph with solid fill")
    addopt(g_style, '--stack', action="store_true", default=False,
           help="stack histograms")
    addopt(g_style, '-m', '--markers', action='store_true', default=False,
           help="add markers to histograms")
    addopt(g_style, '--xerr', action="store_true", default=False,
           help="show width of bins")
    addopt(g_style, '--data', type='int', default=0, metavar='NUM',
           root=opt(help="treat the NUMth target (starting from 1) "
                    "specially, drawing it as black datapoints; to achieve "
                    "a classic data vs. MC plot, try this along with "
                    "--stack and --fill"),
           mpl=opt(help="treat the NUMth target (starting from 1) "
                   "specially, drawing it as black datapoints; to achieve "
                   "a classic data vs. MC plot, try this along with --stack"))
    addopt(g_style, '--xmax', type='float', default=None,
           help="set the maximum value of the x-axis")
    addopt(g_style, '--xmin', type='float', default=None,
           help="set the minimum value of the x-axis")
    addopt(g_style, '--ymax', type='float', default=None,
           help="set the maximum value of the y-axis")
    addopt(g_style, '--ymin', type='float', default=None,
           help="set the minimum value of the y-axis")
    addopt(g_style, '--legend-location', metavar='upper right', default=1,
           help="Place legend in LOC, according to matplotlib "
           "location codes; try 'lower left' or 'center'; "
           "to turn off, set to 'None'")
    addopt(g_style, '--legend-entries', default=None, metavar="''",
           help="A comma-separated string giving the labels to "
           "appear in the legend")
    if use_mpl:
        addopt(g_style, '--legend-ncols', default=None, metavar=1,
               help="Number of columns to use in the legend")
    addopt(g_style, '--title', default=None,
                      help="replace the plot titles, or append to them by "
                      "preceeding with a '+'")
    addopt(g_style, '--xlabel', default=None,
                      help="replace the x-axis labels, or append to them by "
                      "preceeding with a '+'")
    addopt(g_style, '--ylabel', default=None,
                      help="replace the y-axis labels, or append to them by "
                      "preceeding with a '+'")
    addopt(g_style, '--grid', action='store_true', default=False,
                      help="toggle the grid on or off for both axes")
    addopt(g_style, '--gridx', action='store_true', default=False,
                      help="toggle the grid on or off for the x axis")
    addopt(g_style, '--gridy', action='store_true', default=False,
                      help="toggle the grid on or off for the y axis")
    if use_mpl:
        addopt(g_style, '--cmap',
               help="matplotlib colormap to use for 2D plots")
        addopt(g_style, '--barwidth', metadefault=1.0, type=float,
               help="fraction of the bin width for bars to fill")
        addopt(g_style, '--alpha', type='float', metadefault=0.5,
               help="set the opacity of fills")
    addopt(g_style, '--logx', action='store_true', default=False,
           help="force log scale for x axis")
    addopt(g_style, '--logy', action='store_true', default=False,
           help="force log scale for y axis")
    addopt(g_style, '--overflow', action='store_true', default=False,
           help="display overflow content in the highest bin")
    addopt(g_style, '--underflow', action='store_true', default=False,
           help="display underflow content in the lowest bin")
    #### Do immediate processing of arguments
    options, arguments = parser.parse_args(list(argv))
    options = Options(options, arguments, scope=scope)
    options.replace = [] # must have default in case not using mpl
    if options.processors == 1 or options.ext == 'C':
        global use_multiprocessing
        use_multiprocessing = False
    if options.merge: options.ext = 'pdf'
    return options
