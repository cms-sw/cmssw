#!/usr/bin/env python

######################################################################
## File: create_public_lumi_plots.py
######################################################################

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

import numpy as np
from colorsys import hls_to_rgb, rgb_to_hls

import matplotlib
from matplotlib import pyplot as plt
from matplotlib._png import read_png
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.font_manager import FontProperties

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

FONT_PROPS_SUPTITLE = FontProperties(size="x-large", weight="bold")
FONT_PROPS_TITLE = FontProperties(size="medium", weight="bold")
FONT_PROPS_AX_TITLE = FontProperties(size="large", weight="bold")
FONT_PROPS_TICK_LABEL = FontProperties(size="medium", weight="bold")

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
        "b^{-1}" : 1.,
        "Hz/fb" : 1.e-15,
        "Hz/pb" : 1.e-12,
        "Hz/nb" : 1.e-9,
        "Hz/ub" : 1.e-6,
        "Hz/mb" : 1.e-3,
        "Hz/b" : 1.
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

    def max_inst_lum(self, units="Hz/b"):
        res = 0.
        if len(self.data_points):
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

    # End of class LumiDataBlock.

######################################################################

class ColorScheme(object):
    """A bit of a cludge, but a simple way to store color choices."""

    @classmethod
    def InitColors(cls):

        #------------------------------
        # For color scheme 'Greg'.
        #------------------------------

        # This is the light blue of the CMS logo.
        ColorScheme.cms_blue = (0./255., 152./255., 212./255.)

        # This is the orange from the CMS logo.
        ColorScheme.cms_orange = (241./255., 194./255., 40./255.)

        # Slightly darker versions of the above colors for the lines.
        ColorScheme.cms_blue_dark = (102./255., 153./255., 204./255.)
        ColorScheme.cms_orange_dark = (255./255., 153./255., 0./255.)

        #------------------------------
        # For color scheme 'Joe'.
        #------------------------------

        # Several colors from the alternative CMS logo, with their
        # darker line variants.

        ColorScheme.cms_red = (208./255., 0./255., 37./255.)
        ColorScheme.cms_yellow = (255./255., 248./255., 0./255.)
        ColorScheme.cms_purple = (125./255., 16./255., 123./255.)
        ColorScheme.cms_green = (60./255., 177./255., 110./255.)
        ColorScheme.cms_orange2 = (227./255., 136./255., 36./255.)

        # End of InitColors().

    def __init__(self, name):

        self.name = name

        # Some defaults.
        self.color_fill_del = "black"
        self.color_fill_rec = "white"
        self.color_fill_peak = "black"
        self.color_line_del = DarkenColor(self.color_fill_del)
        self.color_line_rec = DarkenColor(self.color_fill_rec)
        self.color_line_peak = DarkenColor(self.color_fill_peak)
        self.logo_name = "cms_logo_1.png"

        tmp_name = self.name.lower()
        if tmp_name == "greg":
            # Color scheme 'Greg'.
            self.color_fill_del = ColorScheme.cms_blue
            self.color_fill_rec = ColorScheme.cms_orange
            self.color_fill_peak = ColorScheme.cms_orange
            self.color_line_del = DarkenColor(self.color_fill_del)
            self.color_line_rec = DarkenColor(self.color_fill_rec)
            self.color_line_peak = DarkenColor(self.color_fill_peak)
            self.logo_name = "cms_logo_2.png"
        elif tmp_name == "joe":
            # Color scheme 'Joe'.
            self.color_fill_del = ColorScheme.cms_yellow
            self.color_fill_rec = ColorScheme.cms_red
            self.color_fill_peak = ColorScheme.cms_red
            self.color_line_del = DarkenColor(self.color_fill_del)
            self.color_line_rec = DarkenColor(self.color_fill_rec)
            self.color_line_peak = DarkenColor(self.color_fill_peak)
            self.logo_name = "cms_logo_3.png"
        else:
            print >> sys.stderr, \
                  "ERROR Unknown color scheme '%s'" % self.name
            sys.exit(1)

        self.file_suffix = "_%s" % tmp_name.lower()

        # End of __init__().

    # End of class ColorScheme.

######################################################################

def CacheFilePath(cache_file_dir, day=None):
    cache_file_path = os.path.abspath(cache_file_dir)
    if day:
        cache_file_name = "lumicalc_cache_%s.csv" % day.isoformat()
        cache_file_path = os.path.join(cache_file_dir, cache_file_name)
    return cache_file_path

######################################################################

def InitMatplotlib():
    """Just some Matplotlib settings."""
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["savefig.dpi"] = 600
    # End of InitMatplotlib().

######################################################################

def DarkenColor(color_in):
    """Takes a tuple (r, g, b) as input."""

    color_tmp = matplotlib.colors.colorConverter.to_rgb(color_in)

    tmp = rgb_to_hls(*color_tmp)
    color_out = hls_to_rgb(tmp[0], .7 * tmp[1], tmp[2])

    # End of DarkenColor().
    return color_out

######################################################################

def AddLogo(logo_name, ax):
    """Read logo from PNG file and add it to axes."""

    logo_data = read_png(logo_name)
    logo_box = OffsetImage(logo_data, zoom=1.2)
    ann_box = AnnotationBbox(logo_box, [0., 1.],
                             xybox=(2., -2.),
                             xycoords="axes fraction",
                             boxcoords="offset points",
                             box_alignment=(0., 1.),
                             pad=0., frameon=False)
    ax.add_artist(ann_box)
    # End of AddLogo().

######################################################################

def GetXLocator(ax):
    """Pick a DateLocator based on the range of the x-axis."""
    (x_lo, x_hi) = ax.get_xlim()
    num_days = x_hi - x_lo
    min_num_ticks = min(num_days, 5)
    locator = matplotlib.dates.AutoDateLocator(minticks=min_num_ticks,
                                               maxticks=None)
    # End of GetLocator().
    return locator

######################################################################

def TweakPlot(fig, ax, (time_begin, time_end),
              add_extra_head_room=False):

    # Add the logo.
    AddLogo(logo_name, ax)

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
        tmp = y_ticks[1] - y_ticks[0]
        (y_min, y_max) = ax.get_ylim()
        ax.set_ylim(y_min, y_max + tmp)

    # Add a second vertical axis on the right-hand side.
    ax_sec = ax.twinx()
    ax_sec.set_ylim(ax.get_ylim())

    for ax_tmp in fig.axes:
        for sub_ax in [ax_tmp.xaxis, ax_tmp.yaxis]:
            for label in sub_ax.get_ticklabels():
                label.set_font_properties(FONT_PROPS_TICK_LABEL)

    time_lo = datetime.datetime.combine(time_begin.date(), datetime.time()) - \
              datetime.timedelta(days=.5)
    time_hi = datetime.datetime.combine(time_end.date(), datetime.time()) + \
              datetime.timedelta(days=.5)
    ax.set_xlim(time_lo, time_hi)

    locator = GetXLocator(ax)
    ax.xaxis.set_major_locator(locator)
    formatter = matplotlib.dates.DateFormatter(DATE_FMT_STR_AXES)
    ax.xaxis.set_major_formatter(formatter)

    fig.subplots_adjust(top=.89, bottom=.125, left=.1, right=.925)
    # End of TweakPlot().

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
        "date_end" : None,
        "color_schemes" : "Joe, Greg",
        "verbose" : False
        }
    cfg_parser = ConfigParser.SafeConfigParser(cfg_defaults)
    cfg_parser.read(config_file_name)

    # Which color scheme to use for drawing the plots.
    color_scheme_names_tmp = cfg_parser.get("general", "color_schemes")
    color_scheme_names = [i.strip() for i in color_scheme_names_tmp.split(",")]
    # Where to store cache files containing the lumiCalc output.
    cache_file_dir = cfg_parser.get("general", "cache_dir")
    # Flag to turn on verbose output.
    verbose = cfg_parser.getboolean("general", "verbose")

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
    cms_energy_str = "%.0f TeV" % (1.e-3 * cms_energy)

    ##########

    # Tell the user what's going to happen.
    print "Using configuration from file '%s'" % config_file_name
    if ignore_cache:
        print "Ignoring all cached lumiCalc results (and rebuilding the cache)"
    else:
        print "Using cached lumiCalc results from %s" % \
              CacheFilePath(cache_file_dir)
    print "Using color schemes '%s'" % ", ".join(color_scheme_names)
    print "Using lumiCalc script '%s'" % lumicalc_script
    print "Using additional lumiCalc flags from configuration: '%s'" % \
          lumicalc_flags_from_cfg
    print "Using overall combination of lumicalc_flags: '%s'" % \
          lumicalc_flags
    print "Selecting data for beam energy %.0f GeV" % beam_energy
    print "Selecting data for accelerator mode '%s'" % accel_mode

    ##########

    # See if the cache file dir exists, otherwise try to create it.
    path_name = CacheFilePath(cache_file_dir)
    if not os.path.exists(path_name):
        if verbose:
            print "Cache file path does not exist: creating it"
        try:
            os.makedirs(path_name)
        except Exception, err:
            print >> sys.stderr, \
                  "ERROR Could not create cache dir: %s" % path_name
            sys.exit(1)

    ##########

    InitMatplotlib()

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
    while day_cur <= date_end:
        year = day_cur.isocalendar()[0]
        week = day_cur.isocalendar()[1]
        weeks.append((year, week))
        day_cur += datetime.timedelta(days=7)
    if num_days <= 7:
        year = date_end.isocalendar()[0]
        week = date_end.isocalendar()[1]
        weeks.append((year, week))

    # Figure out the last day we want to read back from the cache.
    # NOTE: The above checking ensures that date_end is <= today, so
    # the below only assumes that we're never more than two days
    # behind on our luminosity numbers.
    last_day_from_cache = min(today - datetime.timedelta(days=2), date_end)
    if verbose:
        print "Last day for which the cache will be used: %s" % \
              last_day_from_cache.isoformat()

    # First run lumiCalc. Once for each day to be included in the
    # plots.
    print "Running lumiCalc for all requested days"
    for day in days:
        print "  %s" % day.isoformat()
        use_cache = (not ignore_cache) and (day <= last_day_from_cache)
        cache_file_path = CacheFilePath(cache_file_dir, day)
        if (not os.path.exists(cache_file_path)) or (not use_cache):
            date_begin_str = day.strftime(DATE_FMT_STR_LUMICALC)
            date_end_str = (day + datetime.timedelta(days=1)).strftime(DATE_FMT_STR_LUMICALC)
            cmd = "%s --begin '%s' --end '%s' -o %s" % \
                  (lumicalc_cmd, date_begin_str, date_end_str, cache_file_path)
            if verbose:
                print "    running lumicalc as '%s'" % cmd
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
                        if verbose:
                            print "No lumi data for %s, " \
                                  "writing dummy cache file to avoid re-querying the DB" % \
                                  day.isoformat()
                        dummy_file = open(cache_file_path, "w")
                        dummy_file.close()
                else:
                    print >> sys.stderr, \
                          "ERROR Problem running lumiCalc: %s" % output
                    sys.exit(1)
        else:
            if verbose:
                print "    cache file for %s exists" % day.isoformat()

    # Now read back all lumiCalc results.
    print "Reading back lumiCalc results"
    lumi_data_by_day = {}
    for day in days:
        print "  %s" % day.isoformat()
        cache_file_path = CacheFilePath(cache_file_dir, day)
        lumi_data_day = LumiDataBlock()
        try:
            in_file = open(cache_file_path)
            lines = in_file.readlines()
            if not len(lines):
                if verbose:
                    print "    skipping empty file for %s" % day.isoformat()
            else:
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
    ColorScheme.InitColors()

    #------------------------------
    # Create the delivered-lumi plots, one version of everything for
    # each year.
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

        # Figure out the time window of the data included for the plot
        # subtitles.
        time_begin = min([i[1].time_begin() for i in tmp_this_year])
        time_end = max([i[1].time_end() for i in tmp_this_year])
        str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
        str_end = time_end.strftime(DATE_FMT_STR_OUT)

        #----------

        # Build the histograms.
        bin_edges = np.linspace(matplotlib.dates.date2num(day_lo),
                                matplotlib.dates.date2num(day_hi),
                                len(tmp_dates) + 1)
        bin_centers = [.5 * (i + j) for (i, j) \
                       in zip(bin_edges[:-1], bin_edges[1:])]
        # Delivered and recorded luminosity integrated per day.
        units = "pb^{-1}"
        weights_del = [i[1].lum_del_tot(units) for i in tmp_this_year]
        weights_rec = [i[1].lum_rec_tot(units) for i in tmp_this_year]
        # Maximum instantaneous delivered luminosity per day.
        units = "Hz/nb"
        weights_del_inst = [i[1].max_inst_lum(units) for i in tmp_this_year]

        #----------

        # Loop over all color schemes.
        for color_scheme_name in color_scheme_names:

            print "  color scheme '%s'" % color_scheme_name

            color_scheme = ColorScheme(color_scheme_name)
            color_fill_del = color_scheme.color_fill_del
            color_fill_rec = color_scheme.color_fill_rec
            color_fill_peak = color_scheme.color_fill_peak
            color_line_del = color_scheme.color_line_del
            color_line_rec = color_scheme.color_line_rec
            color_line_peak = color_scheme.color_line_peak
            logo_name = color_scheme.logo_name
            file_suffix = color_scheme.file_suffix

            fig = plt.figure()

            #----------

            fig.clear()
            ax = fig.add_subplot(111)

            units = "Hz/nb"

            # Figure out the maximum instantaneous luminosity.
            max_inst = max(weights_del_inst)

            if sum(weights_del) > 0:

                ax.hist(bin_centers, bin_edges, weights=weights_del_inst,
                        histtype="stepfilled",
                        facecolor=color_fill_peak, edgecolor=color_line_peak,
                        label="Max. inst. lumi.: $%.2f$ $\mathrm{%s}$" % \
                        (max_inst, units))

                tmp_leg = ax.legend(loc="upper left",
                                    bbox_to_anchor=(0.025, 0., 1., .97),
                                    frameon=False)
                tmp_leg.legendHandles[0].set_visible(False)

                # Set titles and labels.
                fig.suptitle(r"CMS Peak Luminosity Per Day, " \
                             "%s, %d, $\mathbf{\sqrt{s} = %.0f}$ TeV" % \
                             (particle_type_str, year, 1.e-3 * cms_energy),
                             fontproperties=FONT_PROPS_SUPTITLE)
                ax.set_title("Data included from %s to %s UTC" % \
                             (str_begin, str_end),
                             fontproperties=FONT_PROPS_TITLE)
                ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                ax.set_ylabel(r"Peak Delivered Luminosity ($\mathrm{%s}$)" % units,
                              fontproperties=FONT_PROPS_AX_TITLE)

                TweakPlot(fig, ax, (time_begin, time_end), True)

            fig.savefig("peak_lumi_per_day_%s_%d%s.png" % \
                        (particle_type_str, year, file_suffix))

            #----------

            # The lumi-by-day plot.
            fig.clear()
            ax = fig.add_subplot(111)

            units = "pb^{-1}"

            # Figure out the maximum delivered and recorded luminosities.
            max_del = max(weights_del)
            max_rec = max(weights_rec)

            if sum(weights_del) > 0:

                ax.hist(bin_centers, bin_edges, weights=weights_del,
                        histtype="stepfilled",
                        facecolor=color_fill_del, edgecolor=color_line_del,
                        label="LHC Delivered, max: $%.1f$ $\mathrm{%s}$/day" % \
                        (max_del, units))
                ax.hist(bin_centers, bin_edges, weights=weights_rec,
                        histtype="stepfilled",
                        facecolor=color_fill_rec, edgecolor=color_line_rec,
                    label="CMS Recorded, max: $%.1f$ $\mathrm{%s}$/day" % \
                        (max_rec, units))
                ax.legend(loc="upper left", bbox_to_anchor=(0.125, 0., 1., 1.01),
                          frameon=False)

                # Set titles and labels.
                fig.suptitle(r"CMS Integrated Luminosity Per Day, " \
                             "%s, %d, $\mathbf{\sqrt{s} = %.0f}$ TeV" % \
                             (particle_type_str, year, 1.e-3 * cms_energy),
                             fontproperties=FONT_PROPS_SUPTITLE)
                ax.set_title("Data included from %s to %s UTC" % \
                             (str_begin, str_end),
                             fontproperties=FONT_PROPS_TITLE)
                ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                ax.set_ylabel(r"Integrated Luminosity ($\mathrm{%s}$/day)" % units,
                              fontproperties=FONT_PROPS_AX_TITLE)

                TweakPlot(fig, ax, (time_begin, time_end), True)

            fig.savefig("int_lumi_per_day_%s_%d%s.png" % \
                        (particle_type_str, year, file_suffix))

            #----------

            # Now for the cumulative plot.
            units = "fb^{-1}"
            weights_del = [1.e-3 * i for i in weights_del]
            weights_rec = [1.e-3 * i for i in weights_rec]

            # Figure out the totals.
            tot_del = sum(weights_del)
            tot_rec = sum(weights_rec)

            fig.clear()
            ax = fig.add_subplot(111)

            if sum(weights_del) > 0:

                ax.hist(bin_centers, bin_edges, weights=weights_del,
                        histtype="stepfilled", cumulative=True,
                        facecolor=color_fill_del, edgecolor=color_line_del,
                        label="LHC Delivered: $%.2f$ $\mathrm{%s}$" % \
                        (tot_del, units))
                ax.hist(bin_centers, bin_edges, weights=weights_rec,
                        histtype="stepfilled", cumulative=True,
                        facecolor=color_fill_rec, edgecolor=color_line_rec,
                        label="CMS Recorded: $%.2f$ $\mathrm{%s}$" % \
                        (tot_rec, units))
                ax.legend(loc="upper left", bbox_to_anchor=(0.125, 0., 1., 1.01),
                          frameon=False)

                # Set titles and labels.
                fig.suptitle(r"CMS Integrated Luminosity, " \
                             r"%s, %d, $\mathbf{\sqrt{s} = %.0f}$ TeV" % \
                             (particle_type_str, year, 1.e-3 * cms_energy),
                             fontproperties=FONT_PROPS_SUPTITLE)
                ax.set_title("Data included from %s to %s UTC" % \
                             (str_begin, str_end),
                             fontproperties=FONT_PROPS_TITLE)
                ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                ax.set_ylabel(r"Total Integrated Luminosity ($\mathbf{\mathrm{%s}}$)" % units,
                              fontproperties=FONT_PROPS_AX_TITLE)

                TweakPlot(fig, ax, (time_begin, time_end))

            fig.savefig("int_lumi_cumulative_%s_%d%s.png" % \
                        (particle_type_str, year, file_suffix))

        #----------

        plt.close()

    ##########

    print "Done"

######################################################################
