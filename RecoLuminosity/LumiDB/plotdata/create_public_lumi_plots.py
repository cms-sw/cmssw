#!/usr/bin/env python

######################################################################
## File: create_public_lumi_plots.py
######################################################################

# TODO TODO TODO
# - Implement lumiCalc filtering on particle type!
# - Implement configuration from file.
# - Implement sorting of LumiDataBlocks.
# - Assert there are no under-/overflows in the created histograms.
# - Clean up output (debug vs. verbose).
# - Implement option to plot all color schemes at once.
# TODO TODO TODO end

import sys
import os
import commands
import time
import datetime
import calendar
import copy
import math
import argparse
import ConfigParser

# Make sure ROOT does not eat the command line parameters. Has to be
# before any other ROOT imports.
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = True

from ROOT import kBlack, kWhite
from ROOT import gSystem
from ROOT import gROOT
from ROOT import gStyle
from ROOT import TCanvas
from ROOT import TLegend
from ROOT import TPaveLabel
from ROOT import TPad
from ROOT import TImage
from ROOT import TH1F
from ROOT import TColor
from ROOT import TGaxis

try:
    import debug_hook
    import pdb
except ImportError:
    pass

######################################################################

# Some global constants. Not nice, but okay.
DATE_FMT_STR_LUMICALC = "%m/%d/%y %H:%M:%S"
DATE_FMT_STR_OUT = "%Y-%m-%d %H:%M"
DATE_FMT_STR_AXES = "%-d %b"
DATE_FMT_STR_CFG = "%Y-%m-%d"
NUM_SEC_IN_LS = 2**18 / 11246.

######################################################################

class LumiDataPoint(object):
    """Holds info from one line of lumiCalc lumibyls output."""

    def __init__(self, line):

        # Decode the comma-separated line from lumiCalc.
        line_split = line.split(",")
        tmp = line_split[0].split(":")
        self.run_number = int(tmp[0])
        self.fill_number = int(tmp[1])
        tmp = line_split[2]
        self.timestamp = datetime.datetime.strptime(tmp, DATE_FMT_STR_LUMICALC)
        # NOTE: Convert from ub^{-1} to b^{-1}.
        scale_factor = 1.e6
        self.lum_del = scale_factor * float(line_split[5])
        self.lum_rec = scale_factor * float(line_split[6])

        # End of __init__().

    # End of class LumiDataPoint.

######################################################################

class LumiDataBlock(object):
    """A supposedly coherent block of LumiDataPoints.

    NOTE: No checks on duplicates, sorting, etc.

    """

    scale_factors = {
        "fb^{-1}" : 1.e-15,
        "pb^{-1}" : 1.e-12,
        "nb^{-1}" : 1.e-9,
        "ub^{-1}" : 1.e-6,
        "mb^{-1}" : 1.e-3,
        "b^{-1}" : 1.
        }

    def __init__(self, data_point=None):
        if not data_point:
            self.data_points = []
        else:
            self.data_points = [data_point]
        # End of __init__().

    def __iadd__(self, other):
        self.data_points.extend(other.data_points)
        # End of __iadd__().
        return self

    def add(self, new_point):
        self.data_points.append(new_point)
        # End of add().

    def copy(self):
        # End of copy().
        return copy.deepcopy(self)

    def lum_del_tot(self, units="b^{-1}"):
        res = sum([i.lum_del for i in self.data_points])
        res *= LumiDataBlock.scale_factors[units]
        # End of lum_del_tot().
        return res

    def lum_rec_tot(self, units="b^{-1}"):
        res = sum([i.lum_rec for i in self.data_points])
        res *= LumiDataBlock.scale_factors[units]
        # End of lum_rec_tot().
        return res

    def max_inst_lum(self, units="b^{-1}"):
        res = max([i.lum_del for i in self.data_points])
        res /= NUM_SEC_IN_LS
        res *= LumiDataBlock.scale_factors[units]
        # End of max_inst_lum().
        return res

    def straighten(self):
        self.data_points.sort()
        # End of straighten().

    # BUG BUG BUG
    # The return values of these in case of empty lists could be
    # considered less-than-ideal...
    def time_begin(self):
        res = datetime.datetime.max
        if len(self.data_points):
            res = min([i.timestamp for i in self.data_points])
        # End of time_begin().
        return res

    def time_end(self):
        res = datetime.datetime.min
        if len(self.data_points):
            res = max([i.timestamp for i in self.data_points])
        # End of time_end().
        return res
    # BUG BUG BUG end

    # End of LumiDataBlock.

######################################################################

class ColorScheme(object):
    """A bit of a cludge, but a simple way to store color choices."""

    kCMSBlue = 1756
    kCMSOrange = 1757
    kCMSBlueD = 1759
    kCMSOrangeD = 1760
    kCMSRed = 1700
    kCMSYellow = 1701
    kCMSPurple = 1702
    kCMSGreen = 1703
    kCMSOrange2 = 1704

    @classmethod
    def InitColors(cls):

        #------------------------------
        # For color scheme 'Greg'.
        #------------------------------

        # This is the light blue of the CMS logo.
        ColorScheme.cmsBlue = TColor(ColorScheme.kCMSBlue, 0./255., 152./255., 212./255.)

        # This is the orange from the CMS logo.
        ColorScheme.cmsOrange = TColor(ColorScheme.kCMSOrange, 241./255., 194./255., 40./255.)

        # Slightly darker versions of the above colors for the lines.
        ColorScheme.cmsBlueD = TColor(ColorScheme.kCMSBlueD, 102./255., 153./255., 204./255.)
        ColorScheme.cmsOrangeD = TColor(ColorScheme.kCMSOrangeD, 255./255., 153./255., 0./255.)

        #------------------------------
        # For color scheme 'Joe'.
        #------------------------------

        # Several colors from the alternative CMS logo, with their
        # darker line variants.

        ColorScheme.cmsRed = TColor(ColorScheme.kCMSRed, 208./255., 0./255., 37./255.)
        ColorScheme.cmsYellow = TColor(ColorScheme.kCMSYellow, 255./255., 248./255., 0./255.)
        ColorScheme.cmsPurple = TColor(ColorScheme.kCMSPurple, 125./255., 16./255., 123./255.)
        ColorScheme.cmsGreen = TColor(ColorScheme.kCMSGreen, 60./255., 177./255., 110./255.)
        ColorScheme.cmsOrange2 = TColor(ColorScheme.kCMSOrange2, 227./255., 136./255., 36./255.)

        # End of InitColors().

    def __init__(self, name):

        self.name = name

        # Some defaults.
        self.kFillColorDelivered = kBlack
        self.kFillColorRecorded = kWhite
        self.kFillColorPeak = kBlack
        self.kLineColorDelivered = TColor.GetColorDark(self.kFillColorDelivered)
        self.kLineColorRecorded = TColor.GetColorDark(self.kFillColorRecorded)
        self.kLineColorPeak = TColor.GetColorDark(self.kFillColorPeak)
        self.logo_name = "cms_logo_1.png"

        tmp_name = self.name.lower()
        if tmp_name == "greg":
            # Color scheme 'Greg'.
            self.kFillColorDelivered = ColorScheme.kCMSBlue
            self.kFillColorRecorded = ColorScheme.kCMSOrange
            self.kFillColorPeak = ColorScheme.kCMSOrange
            self.kLineColorDelivered = TColor.GetColorDark(self.kFillColorDelivered)
            self.kLineColorRecorded = TColor.GetColorDark(self.kFillColorRecorded)
            self.kLineColorPeak = TColor.GetColorDark(self.kFillColorPeak)
            self.logo_name = "cms_logo_2.png"
        elif tmp_name == "joe":
            # Color scheme 'Joe'.
            self.kFillColorDelivered = ColorScheme.kCMSYellow
            self.kFillColorRecorded = ColorScheme.kCMSRed
            self.kFillColorPeak = ColorScheme.kCMSRed
            self.kLineColorDelivered = TColor.GetColorDark(self.kFillColorDelivered)
            self.kLineColorRecorded = TColor.GetColorDark(self.kFillColorRecorded)
            self.kLineColorPeak = TColor.GetColorDark(self.kFillColorPeak)
            self.logo_name = "cms_logo_3.png"
        else:
            print >> sys.stderr, \
                  "ERROR Unknown color scheme '%s'" % self.name
            sys.exit(1)

        self.file_suffix = "_%s" % tmp_name.lower()

        # End of __init__().

######################################################################

def CacheFilePath(day, cache_file_dir):
    cache_file_name = "lumicalc_cache_%s.csv" % day.isoformat()
    cache_file_path = os.path.join(cache_file_dir, cache_file_name)
    return cache_file_path

######################################################################

def DarkenColor(color_in):
    """Takes a tuple (r, g, b) as input."""

    tmp = rgb_to_hls(*color_in)
    color_out = hls_to_rgb(tmp[0], .7 * tmp[1], tmp[2])

    # End of DarkenColor().
    return color_out

######################################################################

def AddLogo(logo_name, ax):
    """Read logo from PNG file and add it to axes."""

    logo_data = read_png(logo_name)
    logo_box = OffsetImage(logo_data, zoom=1.)
    ann_box = AnnotationBbox(logo_box, [0., 1.],
                             xybox=(2., -2.),
                             xycoords="axes fraction",
                             boxcoords="offset points",
                             box_alignment=(0., 1.),
                             pad=0., frameon=False)
    ax.add_artist(ann_box)
    # End of AddLogo().

######################################################################

def TweakPlot(fig, ax):

    # Add the logo.
    AddLogo(logo_name, ax)

    # Fiddle with axes ranges etc.
    ax.relim()
    ax.autoscale_view(False, True, True)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(30.)

    # Add a second vertical axis on the right-hand side.
    ax_sec = ax.twinx()
    ax_sec.set_ylim(ax.get_ylim())

    locator = matplotlib.dates.MonthLocator(range(1, 13))
    ax.xaxis.set_major_locator(locator)
    formatter = matplotlib.dates.DateFormatter(DATE_FMT_STR_AXES)
    ax.xaxis.set_major_formatter(formatter)
    fig.subplots_adjust(bottom=.125, left=.1, right=.925)
    # End of TweakPlot().

######################################################################

def LoadMatplotlib():
    """See if we can find and import matplotlib."""

    print "Loading matplotlib"

    matplotlib_loaded = False
    try:
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib._png import read_png
        from matplotlib.offsetbox import OffsetImage
        from matplotlib.offsetbox import AnnotationBbox
        matplotlib.rcParams["legend.numpoints"] = 1
        matplotlib.rcParams["legend.fontsize"] = "medium"
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["text.latex.preamble"] = "\usepackage{amsmath}"
        matplotlib.rcParams["savefig.dpi"] = 600
        matplotlib_loaded = True
        # The following makes the (now local) imports global.
        # (I know, ugly, and probably only works in CPython.)
        sys._getframe(1).f_globals.update(locals())
    except ImportError:
        print "Could not find matplotib"

    return matplotlib_loaded

######################################################################

def InitROOT():
    print "Setting up ROOT style"
    gSystem.Setenv("TZ","UTC");
    gROOT.SetBatch(True)
    gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;")
    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetTitleFillColor(0)
    gStyle.SetTitleBorderSize(0)
    gStyle.SetHistFillStyle(1001)
    gStyle.SetHistFillColor(51)
    gStyle.SetHistLineWidth(2)
    gStyle.SetFrameFillColor(0)
    gStyle.SetTitleW(0.65)
    gStyle.SetTitleH(0.08)
    gStyle.SetTitleX(0.5)
    gStyle.SetTitleAlign(23)
    gStyle.SetStatW(0.25)
    gStyle.SetStatH(0.2)
    gStyle.SetStatColor(0)
    gStyle.SetEndErrorSize(0)
    gStyle.SetPalette(1)
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    gStyle.SetPadBorderMode(0)
    gStyle.SetPadColor(0)
    gStyle.SetGridStyle(0)
    gStyle.SetLegendBorderSize(0)
    gStyle.SetLegendFillColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetFillStyle(4000)
    # End of InitROOT().

######################################################################

def CreateCanvas():
    canvas = TCanvas("canvas", "canvas", 10, 10, 1800, 1400)
    # End of CreateCanvas().
    return canvas

######################################################################

def CreateLegend(shift_x=0., shift_y=0.):
    width = .5
    height = .1
    min_x = .2 + shift_x
    max_x = min_x + width
    min_y = .78 + shift_y
    max_y = min_y + height
    leg = TLegend(min_x, min_y, max_x, max_y)
    leg.SetFillColor(0)
    # End of CreateLegend().
    return leg

######################################################################

def DrawLogo(canvas, logo_name):
    logo = TImage.Open(logo_name)
    aspectRatio = 1. * canvas.GetWh() / canvas.GetWw()
    min_x = .1005
    max_x = .2
    max_y = .899
    min_y = max_y - (max_x - min_x) / aspectRatio
    p = TPad("p", "p", min_x, min_y, max_x, max_y)
    p.SetMargin(.02, .01, .01, .02)
    p.SetBorderSize(0)
    p.SetFillColor(0)
    canvas.cd()
    p.Draw()
    p.cd()
    logo.Draw()
    # End of DrawLogo().

######################################################################

def CreateDateLabel(canvas, start_time, end_time):
    date_str = "Data included from "
    date_str += start_time
    date_str += " to "
    date_str += end_time
    date_str += " UTC"
    x_offset = .4
    y_lo = .82
    height = .2
    label = TPaveLabel(.5 - x_offset, y_lo,
                       .5 + x_offset, y_lo + height,
                       date_str, "NDC")
    label.SetBorderSize(0)
    label.SetFillColor(0)
    # End of CreateDateLabel().
    return label

######################################################################

def CreateDuplicateYAxis(canvas, ax_ori):
    canvas.Update()
    ax = TGaxis(canvas.GetUxmax(), canvas.GetUymin(),
                canvas.GetUxmax(), canvas.GetUymax(),
                canvas.GetUymin(), canvas.GetUymax(),
                ax_ori.GetNdivisions(), "+L")
    ax.SetLineColor(ax_ori.GetAxisColor())
    ax.SetLabelColor(ax_ori.GetLabelColor())
    ax.SetLabelFont(ax_ori.GetLabelFont())
    ax.SetLabelOffset(ax_ori.GetLabelOffset())
    ax.SetLabelSize(ax_ori.GetLabelSize())
    # End of CreateDuplicateYAxis().
    return ax

######################################################################

if __name__ == "__main__":

    desc_str = "This script creates the official CMS luminosity plots " \
               "based on the output from the lumiCalc family of scripts."
    arg_parser = argparse.ArgumentParser(description=desc_str)
    arg_parser.add_argument("configfile", metavar="CONFIGFILE",
                            help="Configuration file name")
    arg_parser.add_argument("--ignore-cache", action="store_true",
                            help="Ignore all cached lumiCalc results " \
                            "and re-query lumiCalc. " \
                            "(Rebuilds the cache as well.)")
    args = arg_parser.parse_args()
    config_file_name = args.configfile
    ignore_cache = args.ignore_cache

    cfg_defaults = {
        "lumicalc_flags" : "",
        "date_end" : None
        }
    cfg_parser = ConfigParser.SafeConfigParser(cfg_defaults)
    cfg_parser.read(config_file_name)

    # Which color scheme to use for drawing the plots.
    color_scheme_name = cfg_parser.get("general", "color_scheme")
    # Where to store cache files containing the lumiCalc output.
    cache_file_dir = cfg_parser.get("general", "cache_dir")
    # Flag to turn on verbose output.
    debug = cfg_parser.get("general", "verbose")

    # Some details on how to invoke lumiCalc.
    lumicalc_script = cfg_parser.get("general", "lumicalc_script")
    lumicalc_flags_from_cfg = cfg_parser.get("general", "lumicalc_flags")
    beam_energy = float(cfg_parser.get("general", "beam_energy"))
    accel_mode = cfg_parser.get("general", "accel_mode")
    lumicalc_flags = "%s --without-checkforupdate " \
                     "--beamenergy %.0f " \
                     "--beamfluctuation 0.15 " \
                     "--amodetag %s " \
                     "lumibyls" % \
                     (lumicalc_flags_from_cfg, beam_energy, accel_mode)
    lumicalc_flags = lumicalc_flags.strip()
    lumicalc_cmd = "%s %s" % (lumicalc_script, lumicalc_flags)

    # Overall begin and end dates of all data to include.
    tmp = cfg_parser.get("general", "date_begin")
    date_begin = datetime.datetime.strptime(tmp, DATE_FMT_STR_CFG).date()
    tmp = cfg_parser.get("general", "date_end")
    date_end = None
    if tmp:
        date_end = datetime.datetime.strptime(tmp, DATE_FMT_STR_CFG).date()
    # If no end date is given, use today.
    today = datetime.datetime.utcnow().date()
    if not date_end:
        print "No end date given --> using today"
        date_end = today
    # If end date lies in the future, truncate at today.
    if date_end > today:
        print "End date lies in the future --> using today instead"
        date_end = today
    # If end date is before start date, give up.
    if date_end < date_begin:
        print >> sys.stderr, \
              "ERROR End date before begin date (%s < %s)" % \
              (date_end.isoformat(), date_begin.isoformat())
        sys.exit(1)

    ##########

    # Map accelerator modes (as fed to lumiCalc) to particle type
    # strings to be used in plot titles etc.
    particle_type_strings = {
        "PROTPHYS" : "pp",
        "IONPHYS" : "PbPb"
        }
    # BUG BUG BUG
    # Implement the '/nucleon' bit in the title for the HI runs!
    # BUG BUG BUG end
    particle_type_str = particle_type_strings[accel_mode]

    cms_energy = 2. * beam_energy

    ##########

    # Tell the user what's going to happen.
    print "Using configuration from file '%s'" % config_file_name
    if ignore_cache:
        print "Ignoring all cached lumiCalc results (and rebuilding the cache)"
    else:
        print "Using cached lumiCalc results from %s" % cache_file_dir
    print "Using color scheme '%s'" % color_scheme_name
    print "Using lumiCalc script '%s'" % lumicalc_script
    print "Using additional lumiCalc flags from configuration: '%s'" % \
          lumicalc_flags_from_cfg
    print "Using overall combination of lumicalc_flags: '%s'" % \
          lumicalc_flags
    print "Selecting data for beam energy %.0f GeV" % beam_energy
    print "Selecting data for accelerator mode '%s'" % accel_mode

    ##########

    week_begin = date_begin.isocalendar()[1]
    week_end = date_end.isocalendar()[1]
    year_begin = date_begin.isocalendar()[0]
    year_end = date_end.isocalendar()[0]
    print "Building a list of days to include in the plots"
    print "  first day to consider: %s (%d, week %d)" % \
          (date_begin.isoformat(), year_begin, week_begin)
    print "  last day to consider:  %s (%d, week %d)" % \
          (date_end.isoformat(), year_end, week_end)
    num_days = (date_end - date_begin).days + 1
    days = [date_begin + datetime.timedelta(days=i) for i in xrange(num_days)]
    years = xrange(year_begin, year_end + 1)
    weeks = []
    day_cur = date_begin
    while day_cur < date_end:
        year = day_cur.isocalendar()[0]
        week = day_cur.isocalendar()[1]
        weeks.append((year, week))
        day_cur += datetime.timedelta(days=7)

    # Figure out the last day we want to read back from the cache.
    # NOTE: The above checking ensures that date_end is <= today, so
    # the below only assumes that we're never more than two days
    # behind on our luminosity numbers.
    last_day_from_cache = date_end - datetime.timedelta(days=2)
    if debug:
        print "DEBUG Last day for which the cache will be used: %s" % \
              last_day_from_cache.isoformat()

    # First run lumiCalc. Once for each day to be included in the
    # plots.
    print "Running lumiCalc for all requested days"
    for day in days:
        use_cache = (not ignore_cache) and (day <= last_day_from_cache)
        cache_file_path = CacheFilePath(day, cache_file_dir)
        if (not os.path.exists(cache_file_path)) or (not use_cache):
            date_begin_str = day.strftime(DATE_FMT_STR_LUMICALC)
            date_end_str = (day + datetime.timedelta(days=1)).strftime(DATE_FMT_STR_LUMICALC)
            cmd = "%s --begin '%s' --end '%s' -o %s" % \
                  (lumicalc_cmd, date_begin_str, date_end_str, cache_file_path)
            if debug:
                print "DEBUG running lumicalc as '%s'" % cmd
            (status, output) = commands.getstatusoutput(cmd)
            if status != 0:
                # This means 'no qualified data found'.
                if (status >> 8) == 13:
                    # If no data is found it never writes the output
                    # file. So for days without data we would keep
                    # querying the database in vain every time the
                    # script runs. To avoid this we just write a dummy
                    # cache file for such days.
                    if output.find("[INFO] No qualified data found, do nothing") > -1:
                        if debug:
                            print "DEBUG No lumi data for %s, " \
                                  "writing dummy cache file to avoid re-querying the DB" % \
                                  day.isoformat()
                        dummy_file = open(cache_file_path, "w")
                        dummy_file.close()
                else:
                    print >> sys.stderr, \
                          "ERROR Problem running lumiCalc: %s" % output
                    sys.exit(1)
        else:
            if debug:
                print "DEBUG Cache file for %s exists" % day.isoformat()

    # Now read back all lumiCalc results.
    print "Reading back lumiCalc results"
    lumi_data_by_day = {}
    for day in days:
        print "DEBUG JGH %s" % day.isoformat()
        cache_file_path = CacheFilePath(day, cache_file_dir)
        lumi_data_day = LumiDataBlock()
        try:
            in_file = open(cache_file_path)
            lines = in_file.readlines()
            # TODO TODO TODO
            # Need to handle this a little nicer.
            if not len(lines):
                if debug:
                    print "DEBUG JGH Skipping empty file for %s" % day.isoformat()
            else:
                # TODO TODO TODO end
                # DEBUG DEBUG DEBUG
                assert lines[0] == "Run:Fill,LS,UTCTime,Beam Status,E(GeV),Delivered(/ub),Recorded(/ub),avgPU\r\n"
                # DEBUG DEBUG DEBUG end
                for line in lines[1:]:
                    lumi_data_day.add(LumiDataPoint(line))
            in_file.close()
        except IOError, err:
            print >> sys.stderr, \
                  "ERROR Could not read lumiCalc results from file '%s': %s" % \
                  (cache_file_path, str(err))
            sys.exit(1)
        lumi_data_by_day[day] = lumi_data_day

    ##########

    # WARNING: This is a bit of a trick. Trim off the front and tail
    # ends without any delivered luminosity. Not sure this is really
    # what we want.

    # Trimming at the start.

    # Find the first day with non-zero delivered luminosity.
    # TODO TODO TODO

    # Trimming at the end.
    # TODO TODO TODO

    ##########

    # Bunch lumiCalc data together into weeks.
    print "Combining lumiCalc data week-by-week"
    lumi_data_by_week = {}
    for (day, lumi) in lumi_data_by_day.iteritems():
        year = day.isocalendar()[0]
        week = day.isocalendar()[1]
        try:
            lumi_data_by_week[year][week] += lumi
        except KeyError:
            try:
                lumi_data_by_week[year][week] = lumi.copy()
            except KeyError:
                lumi_data_by_week[year] = {week: lumi.copy()}

    # Bunch lumiCalc data together into years.
    print "Combining lumiCalc data year-by-year"
    lumi_data_by_year = {}
    for (day, lumi) in lumi_data_by_day.iteritems():
        year = day.isocalendar()[0]
        try:
            lumi_data_by_year[year] += lumi
        except KeyError:
            lumi_data_by_year[year] = lumi.copy()

    ##########

    # Now dump a lot of info to the user.
    sep_line = 50 * "-"
    print sep_line
    units = "pb^{-1}"
    print "Delivered lumi day-by-day (%s):" % units
    print sep_line
    for day in days:
        print "  %s: %5.1f" % \
              (day.isoformat(), lumi_data_by_day[day].lum_del_tot(units))
    print sep_line
    units = "pb^{-1}"
    print "Delivered lumi week-by-week (%s):" % units
    print sep_line
    for (year, week) in weeks:
        print "  %d-%2d: %6.1f" % \
              (year, week, lumi_data_by_week[year][week].lum_del_tot(units))
    print sep_line
    units = "fb^{-1}"
    print "Delivered lumi year-by-year (%s):" % units
    print sep_line
    for year in years:
        print "  %4d: %5.2f" % \
              (year, lumi_data_by_year[year].lum_del_tot(units))
    print sep_line

    ##########

    # And this is where the plotting starts.
    print "Drawing things..."

    InitROOT()
    ColorScheme.InitColors()

    # TODO TODO TODO
    # Should really be moved somewhere else...

    color_scheme = ColorScheme(color_scheme_name)
    kFillColorDelivered = color_scheme.kFillColorDelivered
    kFillColorRecorded = color_scheme.kFillColorRecorded
    kFillColorPeak = color_scheme.kFillColorPeak
    kLineColorDelivered = color_scheme.kLineColorDelivered
    kLineColorRecorded = color_scheme.kLineColorRecorded
    kLineColorPeak = color_scheme.kLineColorPeak
    logo_name = color_scheme.logo_name
    file_suffix = color_scheme.file_suffix
    # TODO TODO TODO end

    #------------------------------
    # Create the delivered-lumi-per-day plots, one for each year.
    #------------------------------

    for year in years:

        tmp_this_year = [(i, j) for (i, j) in lumi_data_by_day.iteritems() \
                         if (i.isocalendar()[0] == year)]
        # TODO TODO TODO
        # Think about this!
        tmp_this_year.sort()
        # TODO TODO TODO end
        tmp_dates = [i[0] for i in tmp_this_year]
        # NOTE: Tweak the time range a bit to force the bins to be
        # drawn from midday to midday.
        day_lo = datetime.datetime.combine(min(tmp_dates), datetime.time()) - \
                 datetime.timedelta(seconds=12*60*60)
        day_hi = datetime.datetime.combine(max(tmp_dates), datetime.time()) + \
                 datetime.timedelta(seconds=12*60*60)
#         bin_edge_lo = time.mktime(day_lo.timetuple())
#         bin_edge_hi = time.mktime(day_hi.timetuple())
        bin_edge_lo = calendar.timegm(day_lo.utctimetuple())
        bin_edge_hi = calendar.timegm(day_hi.utctimetuple())
        h_lum_del = TH1F("h_lum_del", "h_lum_del", len(tmp_dates), bin_edge_lo, bin_edge_hi)
        h_lum_rec = TH1F("h_lum_rec", "h_lum_rec", len(tmp_dates), bin_edge_lo, bin_edge_hi)
        #----------

        # First the lumi-by-day plot.
        units = "pb^{-1}"

        h_lum_del.SetLineColor(kLineColorDelivered)
        h_lum_del.SetMarkerColor(kLineColorDelivered)
        h_lum_del.SetFillColor(kFillColorDelivered)

        h_lum_rec.SetLineColor(kLineColorRecorded)
        h_lum_rec.SetMarkerColor(kLineColorRecorded)
        h_lum_rec.SetFillColor(kFillColorRecorded)

        h_lum_del.SetLineWidth(2)
        h_lum_rec.SetLineWidth(2)

        # Titles etc.

        # Beam energy (since it's also used when calling lumiCalc) is
        # defined in GeV, but we want TeV in the plot.
        cms_energy_str = "%.0f TeV" % (1.e-3 * cms_energy)
        title_per_day = "CMS Integrated Luminosity Per Day, " \
                        "%s, %d, #sqrt{s} = %s;" \
                        "Date;" \
                        "Integrated Luminosity (%s/day)" % \
                        (particle_type_str, year, cms_energy_str, units)
        h_lum_del.SetTitle(title_per_day)
        h_lum_del.GetXaxis().SetTimeDisplay(1)
        h_lum_del.GetXaxis().SetTimeFormat("%d/%m")
        h_lum_del.GetXaxis().SetTimeOffset(0, "gmt")
        h_lum_del.GetXaxis().SetLabelOffset(0.01)
        h_lum_del.GetYaxis().SetTitleOffset(1.2)
        h_lum_del.GetXaxis().SetTitleFont(62)
        h_lum_del.GetYaxis().SetTitleFont(62)
        h_lum_del.GetXaxis().SetNdivisions(705)

        h_lum_del_cum = h_lum_del.__class__(h_lum_del)
        h_lum_rec_cum = h_lum_rec.__class__(h_lum_rec)

        for (bin_num, (day, lumi)) in enumerate(tmp_this_year):
            h_lum_del.SetBinContent(bin_num + 1, lumi.lum_del_tot(units))
            h_lum_rec.SetBinContent(bin_num + 1, lumi.lum_rec_tot(units))

        units = "fb^{-1}"
        title_per_day_cum = "CMS Integrated Luminosity, " \
                            "%s, %d, #sqrt{s} = %s;" \
                            "Date;" \
                            "Integrated Luminosity (%s)" % \
                            (particle_type_str, year, cms_energy_str, units)
        h_lum_del_cum.SetTitle(title_per_day_cum)

        # Figure out the maximum delivered and recorded luminosities.
        max_del = 0.
        max_rec = 0.
        for bin_num in xrange(1, h_lum_del.GetNbinsX() + 1):
            tmp = h_lum_del.GetBinContent(bin_num)
            if tmp > max_del:
                max_del = tmp
            tmp = h_lum_rec.GetBinContent(bin_num)
            if tmp > max_rec:
                max_rec = tmp

        # Tweak the axes ranges a bit to create a bit more 'air.'
        airSpace = .2
        min_y = 0.
        # Round to next multiple of ten.
        tmp = (1. + airSpace) * max(max_del, max_rec)
        max_y = math.ceil(tmp / 10.) * 10
        canvas = CreateCanvas()
        h_lum_del.Draw()
        h_lum_rec.Draw("SAME")
#         h_lum_del.GetYaxis().SetRangeUser(min_y, max_y)

        # Add legend to the 'top left.'
        units = "pb^{-1}"
        legend = CreateLegend()
        marginOld = legend.GetMargin()
        legend.SetX2NDC(legend.GetX2NDC() + 1.01 * (legend.GetX2NDC() - legend.GetX1NDC()))
        legend.SetMargin(marginOld / 1.01)
        legend.AddEntry(h_lum_del,
                        "LHC Delivered, max: %6.1f %s/day" % \
                        (max_del, units),
                        "F")
        legend.AddEntry(h_lum_rec,
                        "CMS Recorded, max: %6.1f %s/day" % \
                        (max_rec, units),
                        "F")
        legend.Draw()

        # Duplicate the vertical axis on the right-hand side.
        ax_sec = CreateDuplicateYAxis(canvas, h_lum_del.GetYaxis())
        ax_sec.Draw()

        # Add a label specifying up until when data taken was included
        # in this plot.
        time_begin = min([i[1].time_begin() for i in tmp_this_year])
        time_end = max([i[1].time_end() for i in tmp_this_year])
        str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
        str_end = time_end.strftime(DATE_FMT_STR_OUT)
        label = CreateDateLabel(canvas, str_begin, str_end)
        label.Draw()

        # Redraw the axes. This way the graphs don't lie on top of the
        # axes any more.
        canvas.RedrawAxis()

        # Add the CMS logo in the top right corner. This has to be the
        # last action so the logo sits on top of the axes.
        DrawLogo(canvas, logo_name)

        canvas.Print("int_lumi_per_day_%s_%d%s.png" % \
                     (particle_type_str, year, file_suffix))

        ##########

        cum_del = 0.
        cum_rec = 0.
        for bin in xrange(h_lum_del.GetNbinsX() + 1):
            cum_del += h_lum_del.GetBinContent(bin)
            h_lum_del_cum.SetBinContent(bin, cum_del)
            cum_rec += h_lum_rec.GetBinContent(bin)
            h_lum_rec_cum.SetBinContent(bin, cum_rec)
        tot_del = 1.e-3 * cum_del
        tot_rec = 1.e-3 * cum_rec

        # Scale to reduce the vertical labels a bit.
        scale_factor = 1.e-3
        h_lum_del_cum.Scale(scale_factor)
        h_lum_rec_cum.Scale(scale_factor)

        # Tweak the axes ranges a bit to create a bit more 'air.'
        airSpace = .2
        min_y = 0.
        # Round to next multiple of ten.
        tmp = (1. + airSpace) * max(tot_del, tot_rec)
        max_y = tmp#math.ceil(tmp / 10.) * 10
        canvas.IsA().Destructor(canvas)
        del canvas
        canvas = CreateCanvas()
        h_lum_del_cum.Draw()
        h_lum_rec_cum.Draw("SAME")
        h_lum_del_cum.GetYaxis().SetRangeUser(min_y, max_y)

        # Add legend to the 'top left.'
        units = "fb^{-1}"
        legend = CreateLegend()
        marginOld = legend.GetMargin()
        legend.SetX2NDC(legend.GetX2NDC() + 1.01 * (legend.GetX2NDC() - legend.GetX1NDC()))
        legend.SetMargin(marginOld / 1.01)
        legend.AddEntry(h_lum_del,
                        "LHC Delivered: %6.2f %s" % \
                        (tot_del, units),
                        "F")
        legend.AddEntry(h_lum_rec,
                        "CMS Recorded: %6.2f %s" % \
                        (tot_rec, units),
                        "F")
        legend.Draw()

        # Duplicate the vertical axis on the right-hand side.
        ax_sec = CreateDuplicateYAxis(canvas, h_lum_del.GetYaxis())
        ax_sec.Draw()

        # Add a label specifying up until when data taken was included
        # in this plot.
        time_begin = min([i[1].time_begin() for i in tmp_this_year])
        time_end = max([i[1].time_end() for i in tmp_this_year])
        str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
        str_end = time_end.strftime(DATE_FMT_STR_OUT)
        label = CreateDateLabel(canvas, str_begin, str_end)
        label.Draw()

        # Redraw the axes. This way the graphs don't lie on top of the
        # axes any more.
        canvas.RedrawAxis()

        # Add the CMS logo in the top right corner. This has to be the
        # last action so the logo sits on top of the axes.
        DrawLogo(canvas, logo_name)

        canvas.Print("int_lumi_cumulative_%s_%d%s.png" % \
                     (particle_type_str, year, file_suffix))

        ##########

        # TODO TODO TODO
        # - Move some of these up.
        import numpy as np
        from colorsys import hls_to_rgb, rgb_to_hls

        # See if we can do the above in Matplotlib as well.
        if not LoadMatplotlib():
            print >> sys.stderr, "ERROR Could not load Matplotlib"
            sys.exit(1)

        color_del_fill = (255./255., 248./255., 0./255.)
        color_rec_fill = (208./255., 0./255., 37./255.)
        color_del_edge = DarkenColor(color_del_fill)
        color_rec_edge = DarkenColor(color_rec_fill)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        units = "pb^{-1}"

        bin_edges = np.linspace(matplotlib.dates.date2num(day_lo),
                                matplotlib.dates.date2num(day_hi),
                                len(tmp_dates) + 1)
        bin_centers = [.5 * (i + j) for (i, j) \
                       in zip(bin_edges[:-1], bin_edges[1:])]
        weights_del = [i[1].lum_del_tot(units) for i in tmp_this_year]
        weights_rec = [i[1].lum_rec_tot(units) for i in tmp_this_year]

        # Figure out the totals.
        # BUG BUG BUG
        # Hmm... ad hoc scaling = not nice.
        tot_del = sum(weights_del) * 1.e-3
        tot_rec = sum(weights_rec) * 1.e-3
        # BUG BUG BUG end

        #----------

        # BUG BUG BUG
        # Should handle this with a bit more grace.
        if sum(weights_del) > 0:
            ax.hist(bin_centers, bin_edges, weights=weights_del,
                    histtype="stepfilled",
                    facecolor=color_del_fill, edgecolor=color_del_edge,
                    label="LHC Delivered, max: $%.1f$ $%s$/day" % \
                    (max_del, units))
            ax.hist(bin_centers, bin_edges, weights=weights_rec,
                    histtype="stepfilled",
                    facecolor=color_rec_fill, edgecolor=color_rec_edge,
                label="CMS Recorded, max: $%.1f$ $%s$/day" % \
                    (max_rec, units))
            ax.legend(loc="upper left", bbox_to_anchor=(0.1, 0., 1., 1.),
                      frameon=False)

            # Set titles and labels.
            fig.suptitle(r"CMS Integrated Luminosity Per Day, " \
                         "%s, %d, $\mathbf{\sqrt{s} = %.0f}$ TeV" % \
                         (particle_type_str, year, 1.e-3 * cms_energy),
                         fontsize=14, fontweight="bold")
            time_begin = min([i[1].time_begin() for i in tmp_this_year])
            time_end = max([i[1].time_end() for i in tmp_this_year])
            str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
            str_end = time_end.strftime(DATE_FMT_STR_OUT)
            ax.set_title("Data included from %s to %s UTC" % \
                         (str_begin, str_end),
                         fontsize=10)
            ax.set_xlabel(r"Date", fontweight="bold")
            ax.set_ylabel(r"Integrated Luminosity ($%s$/day)" % units,
                          fontweight="bold")

            TweakPlot(fig, ax)
            # BUG BUG BUG
            # Bit of magic here.
            y_ticks = ax.get_yticks()
            tmp = y_ticks[1] - y_ticks[0]
            (y_min, y_max) = ax.get_ylim()
            ax.set_ylim(y_min, y_max + tmp)
            # BUG BUG BUG end
            ax.set_xlim(time_begin, time_end)
        # BUG BUG BUG end

        fig.savefig("int_lumi_per_day_%s_%d%s_test.png" % \
                    (particle_type_str, year, file_suffix))

        #----------

        # Now for the cumulative plot.
        units = "fb^{-1}"
        weights_del = [1.e-3 * i for i in weights_del]
        weights_rec = [1.e-3 * i for i in weights_rec]

        fig.clear()
        ax = fig.add_subplot(111)

        # BUG BUG BUG
        # Should handle this with a bit more grace.
        if sum(weights_del) > 0:
            ax.hist(bin_centers, bin_edges, weights=weights_del,
                    histtype="stepfilled", cumulative=True,
                    facecolor=color_del_fill, edgecolor=color_del_edge,
                    label="LHC Delivered: $%.2f$ $%s$" % \
                    (tot_del, units))
            ax.hist(bin_centers, bin_edges, weights=weights_rec,
                    histtype="stepfilled", cumulative=True,
                    facecolor=color_rec_fill, edgecolor=color_rec_edge,
                    label="CMS Recorded: $%.2f$ $%s$" % \
                    (tot_rec, units))
            ax.legend(loc="upper left", bbox_to_anchor=(0.1, 0., 1., 1.),
                      frameon=False)

            # Set titles and labels.
            fig.suptitle(r"CMS Integrated Luminosity, " \
                         "%s, %d, $\mathbf{\sqrt{s} = %.0f}$ TeV" % \
                         (particle_type_str, year, 1.e-3 * cms_energy),
                         fontsize=14, fontweight="bold")
            time_begin = min([i[1].time_begin() for i in tmp_this_year])
            time_end = max([i[1].time_end() for i in tmp_this_year])
            str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
            str_end = time_end.strftime(DATE_FMT_STR_OUT)
            ax.set_title("Data included from %s to %s UTC" % \
                         (str_begin, str_end),
                         fontsize=10)
            ax.set_xlabel(r"Date", fontweight="bold")
            ax.set_ylabel(r"Total Integrated Luminosity ($%s$)" % units,
                          fontweight="bold")

            TweakPlot(fig, ax)
            ax.set_xlim(time_begin, time_end)
        # BUG BUG BUG end

        fig.savefig("int_lumi_cumulative_%s_%d%s_test.png" % \
                    (particle_type_str, year, file_suffix))

        #----------

        plt.close()
        # TODO TODO TODO end

        ##########

        # Elaborate cleanup to avoid trouble with ROOT...
        for i in [h_lum_del, h_lum_rec, legend, canvas]:
            i.IsA().Destructor(i)
            del i

    ##########

    print "Done"

######################################################################
