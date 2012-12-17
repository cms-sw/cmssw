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
import optparse
import ConfigParser

import numpy as np

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

from RecoLuminosity.LumiDB.public_plots_tools import ColorScheme
from RecoLuminosity.LumiDB.public_plots_tools import LatexifyUnits
from RecoLuminosity.LumiDB.public_plots_tools import AddLogo
from RecoLuminosity.LumiDB.public_plots_tools import InitMatplotlib
from RecoLuminosity.LumiDB.public_plots_tools import RoundAwayFromZero
from RecoLuminosity.LumiDB.public_plots_tools import SavePlot
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

# Some global constants. Not nice, but okay.
DATE_FMT_STR_LUMICALC = "%m/%d/%y %H:%M:%S"
DATE_FMT_STR_LUMICALC_DAY = "%m/%d/%y"
DATE_FMT_STR_OUT = "%Y-%m-%d %H:%M"
DATE_FMT_STR_AXES = "%-d %b"
DATE_FMT_STR_CFG = "%Y-%m-%d"
NUM_SEC_IN_LS = 2**18 / 11246.

KNOWN_ACCEL_MODES = ["PROTPHYS", "IONPHYS"]
LEAD_SCALE_FACTOR = 82. / 208.

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

    def __lt__(self, other):
        # End of __lt__().
        return self.time_mid() < other.time_mid()

    def add(self, new_point):
        self.data_points.append(new_point)
        # End of add().

    def copy(self):
        # End of copy().
        return copy.deepcopy(self)

    def is_empty(self):
        # End of is_empty().
        return not len(self.data_points)

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

    def time_begin(self):
        res = min([i.timestamp for i in self.data_points])
        # End of time_begin().
        return res

    def time_end(self):
        res = max([i.timestamp for i in self.data_points])
        # End of time_end().
        return res

    def time_mid(self):
        delta = self.time_end() - self.time_begin()
        delta_sec = delta.days * 24 * 60 * 60 + delta.seconds
        res = self.time_begin() + datetime.timedelta(seconds=.5*delta_sec)
        # End of time_mid().
        return res

    # End of class LumiDataBlock.

######################################################################

class LumiDataBlockCollection(object):
    """A collection of LumiDataBlocks."""

    def __init__(self, data_block=None):
        if not data_block:
            self.data_blocks = []
        else:
            self.data_blocks = [data_block]
        # End of __init__().

    def __len__(self):
        # End of __len__().
        return len(self.data_blocks)

    def add(self, new_block):
        self.data_blocks.append(new_block)
        # End of add().

    def sort(self):
        self.data_blocks.sort()
        # End of sort().

    def time_begin(self):
        res = datetime.datetime.max
        if len(self.data_blocks):
            res = min([i.time_begin() for i in self.data_blocks])
        # End of time_begin().
        return res

    def time_end(self):
        res = datetime.datetime.min
        if len(self.data_blocks):
            res = max([i.time_end() for i in self.data_blocks])
        # End of time_end().
        return res

    def times(self):
        res = [i.time_mid() for i in self.data_blocks]
        # End of times().
        return res

    def lum_del(self, units="b^{-1}"):
        res = [i.lum_del_tot(units) for i in self.data_blocks]
        # End of lum_del().
        return res

    def lum_rec(self, units="b^{-1}"):
        res = [i.lum_rec_tot(units) for i in self.data_blocks]
        # End of lum_rec().
        return res

    def lum_del_tot(self, units="b^{-1}"):
        # End of lum_del().
        return sum(self.lum_del(units))

    def lum_rec_tot(self, units="b^{-1}"):
        # End of lum_rec().
        return sum(self.lum_rec(units))

    def lum_inst_max(self, units="Hz/b"):
        res = [i.max_inst_lum(units) for i in self.data_blocks]
        # End of lum_inst_max().
        return res

    # End of class LumiDataBlockCollection.

######################################################################

def CacheFilePath(cache_file_dir, day=None):
    cache_file_path = os.path.abspath(cache_file_dir)
    if day:
        cache_file_name = "lumicalc_cache_%s.csv" % day.isoformat()
        cache_file_path = os.path.join(cache_file_path, cache_file_name)
    return cache_file_path

######################################################################

def AtMidnight(datetime_in):
    res = datetime.datetime.combine(datetime_in.date(), datetime.time())
    # End of AtMidnight().
    return res

######################################################################

def AtMidWeek(datetime_in):
    tmp = datetime_in.date()
    date_tmp = tmp - \
               datetime.timedelta(days=tmp.weekday()) + \
               datetime.timedelta(days=2)
    res = datetime.datetime.combine(date_tmp, datetime.time())
    # End of AtMidWeek().
    return res

######################################################################

def GetUnits(year, accel_mode, mode):

    units_spec = {
        "PROTPHYS" : {
        2010 : {
        "cum_day" : "pb^{-1}",
        "cum_week" : "pb^{-1}",
        "cum_year" : "pb^{-1}",
        "max_inst" : "Hz/ub",
        },
        2011 : {
        "cum_day" : "pb^{-1}",
        "cum_week" : "pb^{-1}",
        "cum_year" : "fb^{-1}",
        "max_inst" : "Hz/nb",
        },
        2012 : {
        "cum_day" : "pb^{-1}",
        "cum_week" : "fb^{-1}",
        "cum_year" : "fb^{-1}",
        "max_inst" : "Hz/nb",
        }
        },
        "IONPHYS" : {
        2011 : {
        "cum_day" : "ub^{-1}",
        "cum_week" : "ub^{-1}",
        "cum_year" : "ub^{-1}",
        "max_inst" : "Hz/mb",
        }
        }
        }

    units = None

    try:
        units = units_spec[accel_mode][year][mode]
    except KeyError:
        if mode == "cum_day":
            units = "pb^{-1}"
        elif mode == "cum_week":
            units = "pb^{-1}"
        elif mode == "cum_year":
            units = "fb^{-1}"
        elif mode == "max_inst":
            units = "Hz/ub"

    # DEBUG DEBUG DEBUG
    assert not units is None
    # DEBUG DEBUG DEBUG end

    # End of GetUnits().
    return units

######################################################################

def NumDaysInYear(year):
    """Returns the number of days in the given year."""

    date_lo = datetime.date(year, 1, 1)
    date_hi = datetime.date(year + 1, 1, 1)
    num_days = (date_hi - date_lo).days

    # End of NumDaysInYear().
    return num_days

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

    ax.set_xlim(time_begin, time_end)

    locator = GetXLocator(ax)
    ax.xaxis.set_major_locator(locator)
    formatter = matplotlib.dates.DateFormatter(DATE_FMT_STR_AXES)
    ax.xaxis.set_major_formatter(formatter)

    fig.subplots_adjust(top=.85, bottom=.14, left=.13, right=.91)
    # End of TweakPlot().

######################################################################

if __name__ == "__main__":

    desc_str = "This script creates the official CMS luminosity plots " \
               "based on the output from the lumiCalc family of scripts."
    arg_parser = optparse.OptionParser(description=desc_str)
    arg_parser.add_option("--ignore-cache", action="store_true",
                          help="Ignore all cached lumiCalc results " \
                          "and re-query lumiCalc. " \
                          "(Rebuilds the cache as well.)")
    (options, args) = arg_parser.parse_args()
    if len(args) != 1:
        print >> sys.stderr, \
              "ERROR Need exactly one argument: a config file name"
        sys.exit(1)
    config_file_name = args[0]
    ignore_cache = options.ignore_cache

    cfg_defaults = {
        "lumicalc_flags" : "",
        "date_end" : None,
        "color_schemes" : "Joe, Greg",
        "beam_energy" : None,
        "verbose" : False
        }
    cfg_parser = ConfigParser.SafeConfigParser(cfg_defaults)
    if not os.path.exists(config_file_name):
        print >> sys.stderr, \
              "ERROR Config file '%s' does not exist" % config_file_name
        sys.exit(1)
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
    accel_mode = cfg_parser.get("general", "accel_mode")
    # Check if we know about this accelerator mode.
    if not accel_mode in KNOWN_ACCEL_MODES:
        print >> sys.stderr, \
              "ERROR Unknown accelerator mode '%s'" % \
              accel_mode
    beam_energy_tmp = cfg_parser.get("general", "beam_energy")
    # If no beam energy specified, use the default(s) for this
    # accelerator mode.
    beam_energy = None
    beam_energy_from_cfg = None
    if not beam_energy_tmp:
        print "No beam energy specified --> using defaults for '%s'" % \
              accel_mode
        beam_energy_from_cfg = False
    else:
        beam_energy_from_cfg = True
        beam_energy = float(beam_energy_tmp)

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
    particle_type_str = particle_type_strings[accel_mode]

    beam_energy_defaults = {
        "PROTPHYS" : {2010 : 3500.,
                      2011 : 3500.,
                      2012 : 4000.},
        "IONPHYS" : {2010 : 3500.,
                     2011 : 3500.}
        }

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
    print "Selecting data for accelerator mode '%s'" % accel_mode
    if beam_energy_from_cfg:
        print "Selecting data for beam energy %.0f GeV" % beam_energy
    else:
        print "Selecting data for default beam energy for '%s':" % accel_mode
        for (key, val) in beam_energy_defaults[accel_mode].iteritems():
            print "  %d : %.1f GeV" % (key, val)

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
    # DEBUG DEBUG DEBUG
    assert year_end >= year_begin
    # DEBUG DEBUG DEBUG end
    print "Building a list of days to include in the plots"
    print "  first day to consider: %s (%d, week %d)" % \
          (date_begin.isoformat(), year_begin, week_begin)
    print "  last day to consider:  %s (%d, week %d)" % \
          (date_end.isoformat(), year_end, week_end)
    num_days = (date_end - date_begin).days + 1
    days = [date_begin + datetime.timedelta(days=i) for i in xrange(num_days)]
    years = range(year_begin, year_end + 1)
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
        weeks = list(set(weeks))

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
        print "DEBUG JGH %s" % day.isoformat()
        use_cache = (not ignore_cache) and (day <= last_day_from_cache)
        cache_file_path = CacheFilePath(cache_file_dir, day)
        cache_file_tmp = cache_file_path.replace(".csv", "_tmp.csv")
        if (not os.path.exists(cache_file_path)) or (not use_cache):
            date_begin_str = day.strftime(DATE_FMT_STR_LUMICALC)
            date_begin_day_str = day.strftime(DATE_FMT_STR_LUMICALC_DAY)
            date_end_str = (day + datetime.timedelta(days=1)).strftime(DATE_FMT_STR_LUMICALC)
            date_previous_str = (day - datetime.timedelta(days=1)).strftime(DATE_FMT_STR_LUMICALC)
            if not beam_energy_from_cfg:
                year = day.isocalendar()[0]
                beam_energy = beam_energy_defaults[accel_mode][year]
            lumicalc_flags = "%s --without-checkforupdate " \
                             "--beamenergy %.0f " \
                             "--beamfluctuation 0.15 " \
                             "--amodetag %s " \
                             "lumibyls" % \
                             (lumicalc_flags_from_cfg, beam_energy, accel_mode)
            lumicalc_flags = lumicalc_flags.strip()
            lumicalc_cmd = "%s %s" % (lumicalc_script, lumicalc_flags)
            cmd = "%s --begin '%s' --end '%s' -o %s" % \
                  (lumicalc_cmd, date_previous_str, date_end_str, cache_file_tmp)
            if verbose:
                print "    running lumicalc as '%s'" % cmd
            (status, output) = commands.getstatusoutput(cmd)
            # BUG BUG BUG
            # Trying to track down the bad-cache problem.
            output_0 = copy.deepcopy(output)
            # BUG BUG BUG end
            if status != 0:
                # This means 'no qualified data found'.
                if ((status >> 8) == 13 or (status >> 8) == 14):
                    # If no data is found it never writes the output
                    # file. So for days without data we would keep
                    # querying the database in vain every time the
                    # script runs. To avoid this we just write a dummy
                    # cache file for such days.
                    if verbose:
                        print "No lumi data for %s, " \
                              "writing dummy cache file to avoid re-querying the DB" % \
                              day.isoformat()
                    dummy_file = open(cache_file_tmp, "w")
                    dummy_file.write("Run:Fill,LS,UTCTime,Beam Status,E(GeV),Delivered(/ub),Recorded(/ub),avgPU\r\n")
                    dummy_file.close()
                else:
                    print >> sys.stderr, \
                          "ERROR Problem running lumiCalc: %s" % output
                    sys.exit(1)

            # BUG BUG BUG
            # Trying to track down where these empty files come
            # from...
            num_lines_0 = len(open(cache_file_tmp).readlines())
            print "DEBUG JGH Straight after calling lumiCalc: " \
                  "line count in cache file is %d" % num_lines_0
            # BUG BUG BUG end

            # BUG BUG BUG
            # This works around a bug in lumiCalc where sometimes not
            # all data for a given day is returned. The work-around is
            # to ask for data from two days and then filter out the
            # unwanted day.
            lines_to_be_kept = []
            lines_ori = open(cache_file_tmp).readlines()
            for line in lines_ori:
                if (date_begin_day_str in line) or ("Delivered" in line):
                    lines_to_be_kept.append(line)
            newfile = open(cache_file_path, "w")
            newfile.writelines(lines_to_be_kept)
            newfile.close()
            # BUG BUG BUG end

            # BUG BUG BUG
            # Trying to track down where these empty files come
            # from...
            num_lines_1 = len(open(cache_file_path).readlines())
            print "DEBUG JGH After filtering: " \
                  "line count in cache file is %d" % num_lines_1
            # BUG BUG BUG end

            if verbose:
                print "    CSV file for the day written to %s" % \
                      cache_file_path
        else:
            if verbose:
                print "    cache file for %s exists" % day.isoformat()

            # BUG BUG BUG
            # Trying to track down where these empty files come
            # from...
            num_lines_2 = len(open(cache_file_path).readlines())
            print "DEBUG JGH From existing cache file: " \
                  "line count in cache file is %d" % num_lines_2

        # BUG BUG BUG
        # Still trying to track down where these empty files come from.
        date_begin_str = day.strftime(DATE_FMT_STR_LUMICALC)
        date_begin_day_str = day.strftime(DATE_FMT_STR_LUMICALC_DAY)
        date_end_str = (day + datetime.timedelta(days=1)).strftime(DATE_FMT_STR_LUMICALC)
        date_previous_str = (day - datetime.timedelta(days=1)).strftime(DATE_FMT_STR_LUMICALC)
        if not beam_energy_from_cfg:
            year = day.isocalendar()[0]
            beam_energy = beam_energy_defaults[accel_mode][year]
        lumicalc_flags = "%s --without-checkforupdate " \
                         "--beamenergy %.0f " \
                         "--beamfluctuation 0.15 " \
                         "--amodetag %s " \
                         "lumibyls" % \
                         (lumicalc_flags_from_cfg, beam_energy, accel_mode)
        lumicalc_flags = lumicalc_flags.strip()
        lumicalc_cmd = "%s %s" % (lumicalc_script, lumicalc_flags)
        cache_file_dbg = cache_file_path.replace(".csv", "_dbg.csv")
        cmd = "%s --begin '%s' --end '%s' -o %s" % \
              (lumicalc_cmd, date_begin_str, date_end_str, cache_file_dbg)
        print "DEBUG JGH Re-running lumicalc for a single day as '%s'" % cmd
        (status, output) = commands.getstatusoutput(cmd)
        # BUG BUG BUG
        # Trying to track down the bad-cache problem.
        output_1 = copy.deepcopy(output)
        # BUG BUG BUG end
        if status != 0:
            # This means 'no qualified data found'.
            if ((status >> 8) == 13 or (status >> 8) == 14):
                dummy_file = open(cache_file_dbg, "w")
                dummy_file.write("Run:Fill,LS,UTCTime,Beam Status,E(GeV),Delivered(/ub),Recorded(/ub),avgPU\r\n")
                dummy_file.close()
            else:
                print >> sys.stderr, \
                      "ERROR Problem running lumiCalc: %s" % output
                sys.exit(1)
        num_lines_3 = len(open(cache_file_path).readlines())
        num_lines_4 = len(open(cache_file_dbg).readlines())
        print "DEBUG JGH Line count in data file: " \
              "%d" % num_lines_3
        print "DEBUG JGH Line count in cross-check: " \
              "%d" % num_lines_4
        if num_lines_4 != num_lines_3:
            print "DEBUG JGH   !!! Line count mismatch!!!"
            if num_lines_4 < num_lines_3:
                print "DEBUG JGH   !!!  --> found an occurrence of the lumiCalc stoptime bug!!!"
            else:
                print "DEBUG JGH   !!!  --> found a problem !!!"
#                 if output_0:
#                 print "DEBUG JGH   ----------"
#                 print "DEBUG JGH   output from first lumiCalc run:"
#                 print "DEBUG JGH   ----------"
#                 print output_0
#                 print "DEBUG JGH   ----------"
#                 print "DEBUG JGH   output from second lumiCalc run:"
#                 print "DEBUG JGH   ----------"
#                 print output_1
        # BUG BUG BUG end

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
        # Only store data if there actually is something to store.
        if not lumi_data_day.is_empty():
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

    lumi_data_by_week_per_year = {}
    for (year, tmp_lumi) in lumi_data_by_week.iteritems():
        for (week, lumi) in tmp_lumi.iteritems():
            try:
                lumi_data_by_week_per_year[year].add(lumi)
            except KeyError:
                lumi_data_by_week_per_year[year] = LumiDataBlockCollection(lumi)

    # Bunch lumiCalc data together into years.
    print "Combining lumiCalc data year-by-year"
    lumi_data_by_year = {}
    for (day, lumi) in lumi_data_by_day.iteritems():
        year = day.isocalendar()[0]
        try:
            lumi_data_by_year[year] += lumi
        except KeyError:
            lumi_data_by_year[year] = lumi.copy()

    lumi_data_by_day_per_year = {}
    for (day, lumi) in lumi_data_by_day.iteritems():
        year = day.isocalendar()[0]
        try:
            lumi_data_by_day_per_year[year].add(lumi)
        except KeyError:
            lumi_data_by_day_per_year[year] = LumiDataBlockCollection(lumi)

    ##########

    # Now dump a lot of info to the user.
    sep_line = 50 * "-"
    print sep_line
    units = GetUnits(years[-1], accel_mode, "cum_day")
    print "Delivered lumi day-by-day (%s):" % units
    print sep_line
    for day in days:
        tmp_str = "    - (no data, presumably no lumi either)"
        try:
            tmp = lumi_data_by_day[day].lum_del_tot(units)
            helper_str = ""
            if (tmp < .1) and (tmp > 0.):
                helper_str = " (non-zero but very small)"
            tmp_str = "%5.1f%s" % (tmp, helper_str)
        except KeyError:
            pass
        print "  %s: %s" % (day.isoformat(), tmp_str)
    print sep_line
    units = GetUnits(years[-1], accel_mode, "cum_week")
    print "Delivered lumi week-by-week (%s):" % units
    print sep_line
    for (year, week) in weeks:
        tmp_str = "     - (no data, presumably no lumi either)"
        try:
            tmp = lumi_data_by_week[year][week].lum_del_tot(units)
            helper_str = ""
            if (tmp < .1) and (tmp > 0.):
                helper_str = " (non-zero but very small)"
            tmp_str = "%6.1f%s" % (tmp, helper_str)
        except KeyError:
            pass
        print "  %d-%2d: %s" % (year, week, tmp_str)
    print sep_line
    units = GetUnits(years[-1], accel_mode, "cum_year")
    print "Delivered lumi year-by-year (%s):" % units
    print sep_line
    for year in years:
        tmp_str = "    - (no data, presumably no lumi either)"
        try:
            tmp = lumi_data_by_year[year].lum_del_tot(units)
            helper_str = ""
            if (tmp < .01) and (tmp > 0.):
                helper_str = " (non-zero but very small)"
            tmp_str = "%5.2f%s" % (tmp, helper_str)
        except KeyError:
            pass
        print "  %4d: %s" % \
              (year, tmp_str)
    print sep_line

    ##########

    if not len(lumi_data_by_day_per_year):
        print >> sys.stderr, "ERROR No lumi found?"
        sys.exit(1)

    ##########

    # And this is where the plotting starts.
    print "Drawing things..."
    ColorScheme.InitColors()

    #------------------------------
    # Create the per-day delivered-lumi plots.
    #------------------------------

    for year in years:

        print "  daily lumi plots for %d" % year

        if not beam_energy_from_cfg:
            beam_energy = beam_energy_defaults[accel_mode][year]
        cms_energy = 2. * beam_energy
        cms_energy_str = "???"
        if accel_mode == "IONPHYS":
            cms_energy_str = "%.2f TeV/nucleon" % \
                             (1.e-3 * LEAD_SCALE_FACTOR * cms_energy)
        elif accel_mode == "PROTPHYS":
            cms_energy_str = "%.0f TeV" % (1.e-3 * cms_energy)

        lumi_data = lumi_data_by_day_per_year[year]
        lumi_data.sort()

        # NOTE: Tweak the time range a bit to force the bins to be
        # drawn from midday to midday.
        day_lo = AtMidnight(lumi_data.time_begin()) - \
                 datetime.timedelta(seconds=12*60*60)
        day_hi = AtMidnight(lumi_data.time_end()) + \
                 datetime.timedelta(seconds=12*60*60)

        #----------

        # Build the histograms.
        bin_edges = np.linspace(matplotlib.dates.date2num(day_lo),
                                matplotlib.dates.date2num(day_hi),
                                (day_hi - day_lo).days + 1)
        times_tmp = [AtMidnight(i) for i in lumi_data.times()]
        times = [matplotlib.dates.date2num(i) for i in times_tmp]
        # Delivered and recorded luminosity integrated per day.
        units = GetUnits(year, accel_mode, "cum_day")
        weights_del = lumi_data.lum_del(units)
        weights_rec = lumi_data.lum_rec(units)
        # Cumulative versions of the above.
        units = GetUnits(year, accel_mode, "cum_year")
        weights_del_for_cum = lumi_data.lum_del(units)
        weights_rec_for_cum = lumi_data.lum_rec(units)
        # Maximum instantaneous delivered luminosity per day.
        units = GetUnits(year, accel_mode, "max_inst")
        weights_del_inst = lumi_data.lum_inst_max(units)

        # Figure out the time window of the data included for the plot
        # subtitles.
        time_begin = datetime.datetime.combine(lumi_data.time_begin(),
                                               datetime.time()) - \
                                               datetime.timedelta(days=.5)
        time_end = datetime.datetime.combine(lumi_data.time_end(),
                                             datetime.time()) + \
                                             datetime.timedelta(days=.5)
        str_begin = None
        str_end = None
        if sum(weights_del) > 0.:
            str_begin = lumi_data.time_begin().strftime(DATE_FMT_STR_OUT)
            str_end = lumi_data.time_end().strftime(DATE_FMT_STR_OUT)

        #----------

        # Loop over all color schemes.
        for color_scheme_name in color_scheme_names:

            print "    color scheme '%s'" % color_scheme_name

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

            for type in ["lin", "log"]:
                is_log = (type == "log")
                log_setting = False
                if is_log:
                    min_val = min(weights_del_inst)
                    exp = RoundAwayFromZero(math.log10(min_val))
                    log_setting = math.pow(10., exp)

                fig.clear()
                ax = fig.add_subplot(111)

                units = GetUnits(year, accel_mode, "max_inst")

                # Figure out the maximum instantaneous luminosity.
                max_inst = max(weights_del_inst)

                if sum(weights_del) > 0:

                    ax.hist(times, bin_edges, weights=weights_del_inst,
                            histtype="stepfilled",
                            log=log_setting,
                            facecolor=color_fill_peak, edgecolor=color_line_peak,
                            label="Max. inst. lumi.: %.2f %s" % \
                            (max_inst, LatexifyUnits(units)))

                    tmp_leg = ax.legend(loc="upper left",
                                        bbox_to_anchor=(0.025, 0., 1., .97),
                                        frameon=False)
                    tmp_leg.legendHandles[0].set_visible(False)
                    for t in tmp_leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS Peak Luminosity Per Day, " \
                                 "%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                                 (particle_type_str, year, cms_energy_str),
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Peak Delivered Luminosity (%s)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    AddLogo(logo_name, ax)
                    TweakPlot(fig, ax, (time_begin, time_end), True)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "peak_lumi_per_day_%s_%d%s%s" % \
                         (particle_type_str.lower(), year,
                          log_suffix, file_suffix))

            #----------

            # The lumi-per-day plot.
            for type in ["lin", "log"]:
                is_log = (type == "log")
                log_setting = False
                if is_log:
                    min_val = min(weights_rec)
                    exp = RoundAwayFromZero(math.log10(min_val))
                    log_setting = math.pow(10., exp)

                fig.clear()
                ax = fig.add_subplot(111)

                units = GetUnits(year, accel_mode, "cum_day")

                # Figure out the maximum delivered and recorded luminosities.
                max_del = max(weights_del)
                max_rec = max(weights_rec)

                if sum(weights_del) > 0:

                    ax.hist(times, bin_edges, weights=weights_del,
                            histtype="stepfilled",
                            log=log_setting,
                            facecolor=color_fill_del, edgecolor=color_line_del,
                            label="LHC Delivered, max: %.1f %s/day" % \
                            (max_del, LatexifyUnits(units)))
                    ax.hist(times, bin_edges, weights=weights_rec,
                            histtype="stepfilled",
                            log=log_setting,
                            facecolor=color_fill_rec, edgecolor=color_line_rec,
                        label="CMS Recorded, max: %.1f %s/day" % \
                            (max_rec, LatexifyUnits(units)))
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.125, 0., 1., 1.01),
                              frameon=False)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)
                    # Set titles and labels.
                    fig.suptitle(r"CMS Integrated Luminosity Per Day, " \
                                 "%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                                 (particle_type_str, year, cms_energy_str),
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Integrated Luminosity (%s/day)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    AddLogo(logo_name, ax)
                    TweakPlot(fig, ax, (time_begin, time_end), True)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "int_lumi_per_day_%s_%d%s%s" % \
                         (particle_type_str.lower(), year,
                          log_suffix, file_suffix))

            #----------

            # Now for the cumulative plot.
            units = GetUnits(year, accel_mode, "cum_year")

            # Figure out the totals.
            min_del = min(weights_del_for_cum)
            tot_del = sum(weights_del_for_cum)
            tot_rec = sum(weights_rec_for_cum)

            for type in ["lin", "log"]:
                is_log = (type == "log")
                log_setting = False
                if is_log:
                    min_val = min(weights_del_for_cum)
                    exp = RoundAwayFromZero(math.log10(min_val))
                    log_setting = math.pow(10., exp)

                fig.clear()
                ax = fig.add_subplot(111)

                if sum(weights_del) > 0:

                    ax.hist(times, bin_edges, weights=weights_del_for_cum,
                            histtype="stepfilled", cumulative=True,
                            log=log_setting,
                            facecolor=color_fill_del, edgecolor=color_line_del,
                            label="LHC Delivered: %.2f %s" % \
                            (tot_del, LatexifyUnits(units)))
                    ax.hist(times, bin_edges, weights=weights_rec_for_cum,
                            histtype="stepfilled", cumulative=True,
                            log=log_setting,
                            facecolor=color_fill_rec, edgecolor=color_line_rec,
                            label="CMS Recorded: %.2f %s" % \
                            (tot_rec, LatexifyUnits(units)))
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.125, 0., 1., 1.01),
                              frameon=False)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS Integrated Luminosity, " \
                                 r"%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                                 (particle_type_str, year, cms_energy_str),
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Total Integrated Luminosity (%s)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    AddLogo(logo_name, ax)
                    TweakPlot(fig, ax, (time_begin, time_end),
                              add_extra_head_room=is_log)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "int_lumi_per_day_cumulative_%s_%d%s%s" % \
                         (particle_type_str.lower(), year,
                          log_suffix, file_suffix))

    #------------------------------
    # Create the per-week delivered-lumi plots.
    #------------------------------

    for year in years:

        print "  weekly lumi plots for %d" % year

        if not beam_energy_from_cfg:
            beam_energy = beam_energy_defaults[accel_mode][year]
        cms_energy = 2. * beam_energy
        cms_energy_str = "???"
        if accel_mode == "IONPHYS":
            cms_energy_str = "%.2f TeV/nucleon" % \
                             (1.e-3 * LEAD_SCALE_FACTOR * cms_energy)
        elif accel_mode == "PROTPHYS":
            cms_energy_str = "%.0f TeV" % (1.e-3 * cms_energy)

        lumi_data = lumi_data_by_week_per_year[year]
        lumi_data.sort()

        # NOTE: Tweak the time range a bit to force the bins to be
        # split at the middle of the weeks.
        week_lo = AtMidWeek(lumi_data.time_begin()) - \
                  datetime.timedelta(days=3, seconds=12*60*60)
        week_hi = AtMidWeek(lumi_data.time_end()) + \
                  datetime.timedelta(days=3, seconds=12*60*60)

        #----------

        # Build the histograms.
        num_weeks = week_hi.isocalendar()[1] - week_lo.isocalendar()[1] + 1
        bin_edges = np.linspace(matplotlib.dates.date2num(week_lo),
                                matplotlib.dates.date2num(week_hi),
                                num_weeks)
        times = [matplotlib.dates.date2num(i) for i in lumi_data.times()]
        # Delivered and recorded luminosity integrated per week.
        units = GetUnits(year, accel_mode, "cum_week")
        weights_del = lumi_data.lum_del(units)
        weights_rec = lumi_data.lum_rec(units)
        # Cumulative versions of the above.
        units = GetUnits(year, accel_mode, "cum_year")
        weights_del_for_cum = lumi_data.lum_del(units)
        weights_rec_for_cum = lumi_data.lum_rec(units)
        # Maximum instantaneous delivered luminosity per week.
        units = GetUnits(year, accel_mode, "max_inst")
        weights_del_inst = lumi_data.lum_inst_max(units)

        # Figure out the time window of the data included for the plot
        # subtitles.
        str_begin = None
        str_end = None
        if sum(weights_del) > 0.:
            str_begin = lumi_data.time_begin().strftime(DATE_FMT_STR_OUT)
            str_end = lumi_data.time_end().strftime(DATE_FMT_STR_OUT)

        #----------

        # Loop over all color schemes.
        for color_scheme_name in color_scheme_names:

            print "    color scheme '%s'" % color_scheme_name

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

            for type in ["lin", "log"]:
                is_log = (type == "log")
                log_setting = False
                if is_log:
                    min_val = min(weights_del_inst)
                    exp = RoundAwayFromZero(math.log10(min_val))
                    log_setting = math.pow(10., exp)

                fig.clear()
                ax = fig.add_subplot(111)

                units = GetUnits(year, accel_mode, "max_inst")

                # Figure out the maximum instantaneous luminosity.
                max_inst = max(weights_del_inst)

                if sum(weights_del) > 0:

                    ax.hist(times, bin_edges, weights=weights_del_inst,
                            histtype="stepfilled",
                            log=log_setting,
                            facecolor=color_fill_peak, edgecolor=color_line_peak,
                            label="Max. inst. lumi.: %.2f %s" % \
                            (max_inst, LatexifyUnits(units)))

                    tmp_leg = ax.legend(loc="upper left",
                                        bbox_to_anchor=(0.025, 0., 1., .97),
                                        frameon=False)
                    tmp_leg.legendHandles[0].set_visible(False)
                    for t in tmp_leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS Peak Luminosity Per Week, " \
                                 "%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                                 (particle_type_str, year, cms_energy_str),
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)",
                                  fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Peak Delivered Luminosity (%s)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    AddLogo(logo_name, ax)
                    TweakPlot(fig, ax, (week_lo, week_hi), True)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "peak_lumi_per_week_%s_%d%s%s" % \
                         (particle_type_str.lower(), year,
                          log_suffix, file_suffix))

            #----------

            # The lumi-per-week plot.
            for type in ["lin", "log"]:
                is_log = (type == "log")
                log_setting = False
                if is_log:
                    min_val = min(weights_rec)
                    exp = RoundAwayFromZero(math.log10(min_val))
                    log_setting = math.pow(10., exp)

                fig.clear()
                ax = fig.add_subplot(111)

                units = GetUnits(year, accel_mode, "cum_week")

                # Figure out the maximum delivered and recorded luminosities.
                max_del = max(weights_del)
                max_rec = max(weights_rec)

                if sum(weights_del) > 0:

                    ax.hist(times, bin_edges, weights=weights_del,
                            histtype="stepfilled",
                            log=log_setting,
                            facecolor=color_fill_del, edgecolor=color_line_del,
                            label="LHC Delivered, max: %.1f %s/week" % \
                            (max_del, LatexifyUnits(units)))
                    ax.hist(times, bin_edges, weights=weights_rec,
                            histtype="stepfilled",
                            log=log_setting,
                            facecolor=color_fill_rec, edgecolor=color_line_rec,
                        label="CMS Recorded, max: %.1f %s/week" % \
                            (max_rec, LatexifyUnits(units)))
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.125, 0., 1., 1.01),
                              frameon=False)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS Integrated Luminosity Per Week, " \
                                 "%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                                 (particle_type_str, year, cms_energy_str),
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Integrated Luminosity (%s/week)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    AddLogo(logo_name, ax)
                    TweakPlot(fig, ax, (week_lo, week_hi), True)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "int_lumi_per_week_%s_%d%s%s" % \
                         (particle_type_str.lower(), year,
                          log_suffix, file_suffix))

            #----------

            # Now for the cumulative plot.
            units = GetUnits(year, accel_mode, "cum_year")

            # Figure out the totals.
            min_del = min(weights_del_for_cum)
            tot_del = sum(weights_del_for_cum)
            tot_rec = sum(weights_rec_for_cum)

            for type in ["lin", "log"]:
                is_log = (type == "log")
                log_setting = False
                if is_log:
                    min_val = min(weights_del_for_cum)
                    exp = RoundAwayFromZero(math.log10(min_val))
                    log_setting = math.pow(10., exp)

                fig.clear()
                ax = fig.add_subplot(111)

                if sum(weights_del) > 0:

                    ax.hist(times, bin_edges, weights=weights_del_for_cum,
                            histtype="stepfilled", cumulative=True,
                            log=log_setting,
                            facecolor=color_fill_del, edgecolor=color_line_del,
                            label="LHC Delivered: %.2f %s" % \
                            (tot_del, LatexifyUnits(units)))
                    ax.hist(times, bin_edges, weights=weights_rec_for_cum,
                            histtype="stepfilled", cumulative=True,
                            log=log_setting,
                            facecolor=color_fill_rec, edgecolor=color_line_rec,
                            label="CMS Recorded: %.2f %s" % \
                            (tot_rec, LatexifyUnits(units)))
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.125, 0., 1., 1.01),
                              frameon=False)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS Integrated Luminosity, " \
                                 r"%s, %d, $\mathbf{\sqrt{s} =}$ %s" % \
                                 (particle_type_str, year, cms_energy_str),
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Total Integrated Luminosity (%s)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    AddLogo(logo_name, ax)
                    TweakPlot(fig, ax, (week_lo, week_hi),
                              add_extra_head_room=is_log)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "int_lumi_per_week_cumulative_%s_%d%s%s" % \
                            (particle_type_str.lower(), year,
                             log_suffix, file_suffix))

    plt.close()

    #----------

    # Now the cumulative plot showing all years together.
    if len(years) > 1:
        print "  cumulative luminosity for %s together" % ", ".join([str(i) for i in years])

        def PlotAllYears(lumi_data_by_day_per_year, mode):
            """Mode 1: years side-by-side, mode 2: years overlaid."""

            units = GetUnits(years[-1], accel_mode, "cum_year")

            scale_factor_2010 = 100.

            # Loop over all color schemes and plot.
            for color_scheme_name in color_scheme_names:

                print "      color scheme '%s'" % color_scheme_name

                color_scheme = ColorScheme(color_scheme_name)
                color_by_year = color_scheme.color_by_year
                logo_name = color_scheme.logo_name
                file_suffix = color_scheme.file_suffix

                for type in ["lin", "log"]:
                    is_log = (type == "log")

                    if mode == 1:
                        aspect_ratio = matplotlib.figure.figaspect(1. / 2.5)
                        fig = plt.figure(figsize=aspect_ratio)
                    else:
                        fig = plt.figure()
                    ax = fig.add_subplot(111)

                    time_begin_ultimate = lumi_data_by_day_per_year[years[0]].time_begin()
                    str_begin_ultimate = time_begin_ultimate.strftime(DATE_FMT_STR_OUT)
                    for (year_index, year) in enumerate(years):

                        lumi_data = lumi_data_by_day_per_year[year]
                        lumi_data.sort()
                        times_tmp = [AtMidnight(i) for i in lumi_data.times()]
                        # For the plots showing all years overlaid, shift
                        # all but the first year forward.
                        # NOTE: Years list is supposed to be sorted.
                        if mode == 2:
                            if year_index > 0:
                                for y in years[:year_index]:
                                    num_days = NumDaysInYear(y)
                                    time_shift = datetime.timedelta(days=num_days)
                                    times_tmp = [(i - time_shift) \
                                                 for i in times_tmp]
                        times = [matplotlib.dates.date2num(i) for i in times_tmp]
                        # DEBUG DEBUG DEBUG
                        for i in xrange(len(times) - 1):
                            assert times[i] < times[i + 1]
                        # DEBUG DEBUG DEBUG end
                        weights_del = lumi_data.lum_del(units)
                        weights_del_cum = [0.] * len(weights_del)
                        tot_del = 0.
                        for (i, val) in enumerate(weights_del):
                            tot_del += val
                            weights_del_cum[i] = tot_del
                        if not beam_energy_from_cfg:
                            beam_energy = beam_energy_defaults[accel_mode][year]
                        cms_energy = 2. * beam_energy
                        cms_energy_str = "???"
                        if accel_mode == "IONPHYS":
                            cms_energy_str = "%.2f TeV/nucleon" % \
                                             (1.e-3 * LEAD_SCALE_FACTOR * cms_energy)
                        elif accel_mode == "PROTPHYS":
                            cms_energy_str = "%.0f TeV" % (1.e-3 * cms_energy)

                        # NOTE: Special case for 2010.
                        label = None
                        if year == 2010:
                            label = r"%d, %s, %.1f %s" % \
                                    (year, cms_energy_str,
                                     1.e3 * tot_del,
                                     LatexifyUnits("pb^{-1}"))
                        else:
                            label = r"%d, %s, %.1f %s" % \
                                    (year, cms_energy_str, tot_del,
                                     LatexifyUnits(units))

                        # NOTE: Special case for 2010
                        weights_tmp = None
                        if year == 2010:
                            weights_tmp = [scale_factor_2010 * i \
                                           for i in weights_del_cum]
                        else:
                            weights_tmp = weights_del_cum
                        ax.plot(times, weights_tmp,
                                color=color_by_year[year],
                                marker="none", linestyle="solid",
                                linewidth=4,
                                label=label)
                        if is_log:
                            ax.set_yscale("log")

                        # NOTE: Special case for 2010.
                        if year == 2010:
                            ax.annotate(r"$\times$ %.0f" % scale_factor_2010,
                                        xy=(times[-1], weights_tmp[-1]),
                                        xytext=(5., -2.),
                                        xycoords="data", textcoords="offset points")

                    # BUG BUG BUG
                    # Needs work...
                    time_begin = lumi_data.time_begin()
                    time_end = lumi_data.time_end()
                    str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
                    str_end = time_end.strftime(DATE_FMT_STR_OUT)
                    if mode == 1:
                        time_begin = datetime.datetime(years[0], 1, 1, 0, 0, 0)
                        time_end = datetime.datetime(years[-1], 12, 31, 23, 59,59)
                    else:
                        time_begin = datetime.datetime(years[0], 1, 1, 0, 0, 0)
                        time_end = datetime.datetime(years[0], 12, 31, 23, 59,59)
                    # BUG BUG BUG end

                    num_cols = None
                    if mode == 1:
                        num_cols = len(years)
                        tmp_x = 0.095
                        tmp_y = .95
                    else:
                        num_cols = 1
                        tmp_x = 0.175
                        tmp_y = 1.01
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(tmp_x, 0., 1., tmp_y),
                              frameon=False, ncol=num_cols)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS Integrated Luminosity, %s" % particle_type_str,
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
#                                 (str_begin, str_end),
                                 (str_begin_ultimate, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Total Integrated Luminosity (%s)" % \
                                  LatexifyUnits(units),
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    zoom = 1.7
                    if mode == 1:
                        zoom = .95
                    AddLogo(logo_name, ax, zoom=zoom)
                    extra_head_room = 0
                    if is_log:
                        if mode == 1:
                            extra_head_room = 1
                        elif mode == 2:
                            extra_head_room = 2
#                    TweakPlot(fig, ax, (time_begin, time_end),
                    TweakPlot(fig, ax, (time_begin_ultimate, time_end),
                              add_extra_head_room=extra_head_room)

                    log_suffix = ""
                    if is_log:
                        log_suffix = "_log"
                    SavePlot(fig, "int_lumi_cumulative_%s_%d%s%s" % \
                             (particle_type_str.lower(), mode,
                              log_suffix, file_suffix))

        for mode in [1, 2]:
            print "    mode %d" % mode
            PlotAllYears(lumi_data_by_day_per_year, mode)

    plt.close()

    #----------

    # Now the peak lumi plot showing all years together.
    if len(years) > 1:
        print "  peak luminosity for %s together" % ", ".join([str(i) for i in years])

        units = GetUnits(years[-1], accel_mode, "max_inst")

        scale_factor_2010 = 10.

        # Loop over all color schemes and plot.
        for color_scheme_name in color_scheme_names:

            print "      color scheme '%s'" % color_scheme_name

            color_scheme = ColorScheme(color_scheme_name)
            color_by_year = color_scheme.color_by_year
            logo_name = color_scheme.logo_name
            file_suffix = color_scheme.file_suffix

            for type in ["lin", "log"]:
                is_log = (type == "log")

                aspect_ratio = matplotlib.figure.figaspect(1. / 2.5)
                fig = plt.figure(figsize=aspect_ratio)
                ax = fig.add_subplot(111)

                time_begin_ultimate = lumi_data_by_day_per_year[years[0]].time_begin()
                str_begin_ultimate = time_begin_ultimate.strftime(DATE_FMT_STR_OUT)
                for (year_index, year) in enumerate(years):

                    lumi_data = lumi_data_by_day_per_year[year]
                    lumi_data.sort()
                    times_tmp = [AtMidnight(i) for i in lumi_data.times()]
                    times = [matplotlib.dates.date2num(i) for i in times_tmp]
                    # DEBUG DEBUG DEBUG
                    for i in xrange(len(times) - 1):
                        assert times[i] < times[i + 1]
                    # DEBUG DEBUG DEBUG end
                    weights_inst = lumi_data.lum_inst_max(units)
                    max_inst = max(weights_inst)
                    if not beam_energy_from_cfg:
                        beam_energy = beam_energy_defaults[accel_mode][year]
                    cms_energy = 2. * beam_energy
                    cms_energy_str = "???"
                    if accel_mode == "IONPHYS":
                        cms_energy_str = "%.2f TeV/nucleon" % \
                                         (1.e-3 * LEAD_SCALE_FACTOR * cms_energy)
                    elif accel_mode == "PROTPHYS":
                        cms_energy_str = "%.0f TeV" % (1.e-3 * cms_energy)

                    # NOTE: Special case for 2010.
                    label = None
                    if year == 2010:
                        label = r"%d, %s, max. %.1f %s" % \
                                (year, cms_energy_str,
                                 1.e3 * max_inst,
                                 LatexifyUnits("Hz/ub"))
                    else:
                        label = r"%d, %s, max. %.1f %s" % \
                                (year, cms_energy_str, max_inst,
                                 LatexifyUnits(units))

                    # NOTE: Special case for 2010
                    weights_tmp = None
                    if year == 2010:
                        weights_tmp = [scale_factor_2010 * i \
                                       for i in weights_inst]
                    else:
                        weights_tmp = weights_inst
                    ax.plot(times, weights_tmp,
                            color=color_by_year[year],
                            marker=".", markersize=8.,
                            linestyle="none",
                            label=label)
                    if is_log:
                        ax.set_yscale("log")

                    # NOTE: Special case for 2010.
                    if year == 2010:
                        ax.annotate(r"$\times$ %.0f" % scale_factor_2010,
                                    xy=(times[-1], max(weights_tmp)),
                                    xytext=(5., -2.),
                                    xycoords="data", textcoords="offset points")

                    # BUG BUG BUG
                    # Needs work...
                    time_begin = lumi_data.time_begin()
                    time_end = lumi_data.time_end()
                    str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
                    str_end = time_end.strftime(DATE_FMT_STR_OUT)
                    time_begin = datetime.datetime(years[0], 1, 1, 0, 0, 0)
                    time_end = datetime.datetime(years[-1], 12, 31, 23, 59,59)
                    # BUG BUG BUG end

                    num_cols = None
                    num_cols = len(years) - 2
                    tmp_x = .09
                    tmp_y = .97
                    leg = ax.legend(loc="upper left",
                              bbox_to_anchor=(tmp_x, 0., 1., tmp_y),
                              labelspacing=.2,
                              columnspacing=.2,
                              frameon=False, ncol=num_cols)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                # Set titles and labels.
                fig.suptitle(r"CMS Peak Luminosity Per Day, %s" % particle_type_str,
                             fontproperties=FONT_PROPS_SUPTITLE)
                ax.set_title("Data included from %s to %s UTC \n" % \
#                             (str_begin, str_end),
                             (str_begin_ultimate, str_end),
                             fontproperties=FONT_PROPS_TITLE)
                ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                ax.set_ylabel(r"Peak Delivered Luminosity (%s)" % \
                              LatexifyUnits(units),
                              fontproperties=FONT_PROPS_AX_TITLE)

                # Add the logo.
                zoom = .97
                AddLogo(logo_name, ax, zoom=zoom)
                head_room = 2.
                if is_log:
                    head_room = 2.
#                TweakPlot(fig, ax, (time_begin, time_end),
                TweakPlot(fig, ax, (time_begin_ultimate, time_end),
                          add_extra_head_room=head_room)

                log_suffix = ""
                if is_log:
                    log_suffix = "_log"
                SavePlot(fig, "peak_lumi_%s%s%s" % \
                         (particle_type_str.lower(),
                          log_suffix, file_suffix))

    #----------

    plt.close()

    ##########

    print "Done"

######################################################################
