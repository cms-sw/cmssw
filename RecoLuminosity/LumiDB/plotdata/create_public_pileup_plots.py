#!/usr/bin/env python

######################################################################
## File: create_public_pileup_plots.py
######################################################################

# NOTE: Typical way to create the pileup ROOT file from the cached txt
# files (maintained by Mike Hildredth):
# pileupCalc.py -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/DCSOnly/json_DCSONLY.txt \
# --inputLumiJSON=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/PileUp/pileup_latest.txt \
# --calcMode true --maxPileupBin=40 pu2012DCSONLY.root

import sys
import os
import commands
import math
import optparse
import ConfigParser

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# FIX FIX FIX
# This fixes a well-know bug with stepfilled logarithmic histograms in
# Matplotlib.
from RecoLuminosity.LumiDB.mpl_axes_hist_fix import hist
if matplotlib.__version__ != '1.0.1':
    print >> sys.stderr, \
          "ERROR The %s script contains a hard-coded bug-fix " \
          "for Matplotlib 1.0.1. The Matplotlib version loaded " \
          "is %s" % (__file__, matplotlib.__version__)
    sys.exit(1)
matplotlib.axes.Axes.hist = hist
# FIX FIX FIX end

from ROOT import gROOT
gROOT.SetBatch(True)
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = True
from ROOT import TFile

from RecoLuminosity.LumiDB.public_plots_tools import ColorScheme
from RecoLuminosity.LumiDB.public_plots_tools import LatexifyUnits
from RecoLuminosity.LumiDB.public_plots_tools import AddLogo
from RecoLuminosity.LumiDB.public_plots_tools import RoundAwayFromZero
from RecoLuminosity.LumiDB.public_plots_tools import FONT_PROPS_SUPTITLE
from RecoLuminosity.LumiDB.public_plots_tools import FONT_PROPS_TITLE
from RecoLuminosity.LumiDB.public_plots_tools import FONT_PROPS_AX_TITLE
from RecoLuminosity.LumiDB.public_plots_tools import FONT_PROPS_TICK_LABEL

try:
    import debug_hook
    import pdb
except ImportError:
    pass

######################################################################

# Some hard-coded thingies. Not nice, but no time to do this
# differently right now.
particle_type_str = "pp"
year = 2012
cms_energy_str = "8 TeV"

######################################################################

def InitMatplotlib():
    """Just some Matplotlib settings."""
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["figure.figsize"] = (8., 6.)
    matplotlib.rcParams["figure.dpi"] = 200
    matplotlib.rcParams["savefig.dpi"] = matplotlib.rcParams["figure.dpi"]
    # End of InitMatplotlib().

######################################################################

def TweakPlot(fig, ax, add_extra_head_room=False):

    # Fiddle with axes ranges etc.
    ax.relim()
    ax.autoscale_view(False, True, True)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(30.)

    # Bit of magic here: increase vertical scale by one tick to make
    # room for the legend.
    if add_extra_head_room:
        y_ticks = ax.get_yticks()
        (y_min, y_max) = ax.get_ylim()
        is_log = (ax.get_yscale() == "log")
        y_max_new = y_max
        if is_log:
            tmp = y_ticks[-1] / y_ticks[-2]
            y_max_new = y_max * math.pow(tmp, add_extra_head_room)
        else:
            tmp = y_ticks[-1] - y_ticks[-2]
            y_max_new = y_max + add_extra_head_room * tmp
        ax.set_ylim(y_min, y_max_new)

    # Add a second vertical axis on the right-hand side.
    ax_sec = ax.twinx()
    ax_sec.set_ylim(ax.get_ylim())
    ax_sec.set_yscale(ax.get_yscale())

    for ax_tmp in fig.axes:
        for sub_ax in [ax_tmp.xaxis, ax_tmp.yaxis]:
            for label in sub_ax.get_ticklabels():
                label.set_font_properties(FONT_PROPS_TICK_LABEL)

    if is_log:
        fig.subplots_adjust(top=.89, bottom=.125, left=.11, right=.925)
    else:
        fig.subplots_adjust(top=.89, bottom=.125, left=.1, right=.925)

    # End of TweakPlot().

######################################################################

if __name__ == "__main__":

    desc_str = "This script creates the official CMS pileup plots " \
               "based on the output from the pileupCalc.py script."
    arg_parser = optparse.OptionParser(description=desc_str)
    arg_parser.add_option("--ignore-cache", action="store_true",
                          help="Ignore all cached PU results " \
                          "and run pileupCalc. " \
                          "(Rebuilds the cache as well.)")
    (options, args) = arg_parser.parse_args()
    if len(args) != 1:
        print >> sys.stderr, \
              "ERROR Need exactly one argument: a config file name"
        sys.exit(1)
    config_file_name = args[0]
    ignore_cache = options.ignore_cache

    cfg_defaults = {
        "pileupcalc_flags" : "",
        "color_schemes" : "Joe, Greg",
        "verbose" : False
        }
    cfg_parser = ConfigParser.SafeConfigParser(cfg_defaults)
    if not os.path.exists(config_file_name):
        print >> sys.stderr, \
              "ERROR Config file '%s' does not exist" % config_file_name
        sys.exit(1)
    cfg_parser.read(config_file_name)

    # Location of the cached ROOT file.
    cache_file_dir = cfg_parser.get("general", "cache_dir")

    # Which color scheme to use for drawing the plots.
    color_scheme_names_tmp = cfg_parser.get("general", "color_schemes")
    color_scheme_names = [i.strip() for i in color_scheme_names_tmp.split(",")]
    # Flag to turn on verbose output.
    verbose = cfg_parser.getboolean("general", "verbose")

    # Some details on how to invoke pileupCalc.
    pileupcalc_flags_from_cfg = cfg_parser.get("general", "pileupcalc_flags")
    input_json = cfg_parser.get("general", "input_json")
    input_lumi_json = cfg_parser.get("general", "input_lumi_json")

    ##########

    # Tell the user what's going to happen.
    print "Using configuration from file '%s'" % config_file_name
    print "Using color schemes '%s'" % ", ".join(color_scheme_names)
    print "Using additional pileupCalc flags from configuration: '%s'" % \
          pileupcalc_flags_from_cfg
    print "Using input JSON filter: %s" % input_json
    print "Using input lumi JSON filter: %s" % input_lumi_json

    ##########

    InitMatplotlib()

    ##########

    # First run pileupCalc.
    tmp_file_name = os.path.join(cache_file_dir,"pileup_calc_tmp.root")
    if (not ignore_cache):
        cmd = "pileupCalc.py -i %s --inputLumiJSON=%s %s %s" % \
              (input_json, input_lumi_json,
               pileupcalc_flags_from_cfg, tmp_file_name)
        print "Running pileupCalc (this may take a while)"
        if verbose:
            print "  pileupCalc cmd: '%s'" % cmd
        (status, output) = commands.getstatusoutput(cmd)
        if status != 0:
            print >> sys.stderr, \
                  "ERROR Problem running pileupCalc: %s" % output
            sys.exit(1)

    ##########

    in_file = TFile.Open(tmp_file_name, "READ")
    if not in_file or in_file.IsZombie():
        print >> sys.stderr, \
              "ERROR Could not read back pileupCalc results"
        sys.exit(1)
    pileup_hist = in_file.Get("pileup")
    pileup_hist.SetDirectory(0)
    in_file.Close()

    ##########

    # And this is where the plotting starts.
    print "Drawing things..."
    ColorScheme.InitColors()

    # Turn the ROOT histogram into a Matplotlib one.
    bin_edges = [pileup_hist.GetBinLowEdge(i) \
                 for i in xrange(1, pileup_hist.GetNbinsX() + 1)]
    vals = [pileup_hist.GetBinCenter(i) \
            for i in xrange(1, pileup_hist.GetNbinsX() + 1)]
    weights = [pileup_hist.GetBinContent(i) \
               for i in xrange(1, pileup_hist.GetNbinsX() + 1)]
    # NOTE: Convert units to /pb!
    weights = [1.e-6 * i for i in weights]

    # Loop over all color schemes.
    for color_scheme_name in color_scheme_names:

        print "    color scheme '%s'" % color_scheme_name

        color_scheme = ColorScheme(color_scheme_name)
        color_line_pileup = color_scheme.color_line_pileup
        color_fill_pileup = color_scheme.color_fill_pileup
        logo_name = color_scheme.logo_name
        file_suffix = color_scheme.file_suffix

        fig = plt.figure()

        for type in ["lin", "log"]:
            is_log = (type == "log")
            log_setting = False
            if is_log:
                min_val = min(weights)
                exp = RoundAwayFromZero(math.log10(min_val))
                log_setting = math.pow(10., exp)

            fig.clear()
            ax = fig.add_subplot(111)

            ax.hist(vals, bins=bin_edges, weights=weights, log=log_setting,
                    histtype="stepfilled",
                    edgecolor=color_line_pileup,
                    facecolor=color_fill_pileup)

            # Set titles and labels.
            fig.suptitle(r"CMS Average Pileup, " \
                         "%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                         (particle_type_str, year, cms_energy_str),
                         fontproperties=FONT_PROPS_SUPTITLE)
            ax.set_xlabel(r"Mean number of interactions per crossing",
                          fontproperties=FONT_PROPS_AX_TITLE)
            ax.set_ylabel(r"Recorded Luminosity (%s/%.2f)" % \
                          (LatexifyUnits("pb^{-1}"),
                           pileup_hist.GetBinWidth(1)),
                          fontproperties=FONT_PROPS_AX_TITLE)

            # Add the average pileup number to the top right.
            ax.text(.95, .925, r"<$\mathbf{\mu}$> = %.0f" % \
                    round(pileup_hist.GetMean()),
                    transform = ax.transAxes,
                    horizontalalignment="right",
                    fontproperties=FONT_PROPS_AX_TITLE)

            # Add the logo.
            AddLogo(logo_name, ax)
            TweakPlot(fig, ax, True)

            log_suffix = ""
            if is_log:
                log_suffix = "_log"
            fig.savefig("pileup_%s_%d%s%s.png" % \
                        (particle_type_str, year,
                         log_suffix, file_suffix))

        plt.close()

    ##########

    print "Done"

######################################################################
