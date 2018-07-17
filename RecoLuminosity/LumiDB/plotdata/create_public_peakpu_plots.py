#!/usr/bin/env python

######################################################################
## File: create_public_lumi_plots.py
######################################################################

import sys
import csv
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

import six
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

KNOWN_ACCEL_MODES = ["PROTPHYS", "IONPHYS", "PAPHYS",
                     "2013_amode_bug_workaround"]
LEAD_SCALE_FACTOR = 82. / 208.

def processdata(years):
    results={}#{year:[[str,str],[pu,pu]]}
    for y in years:
        fills=[]#[fillnum]
        maxputimelist=[]
        maxpulist=[]
        results[y]=[maxputimelist,maxpulist]
        f=None
        try:
            lumifilename=str(y)+'lumibyls.csv'
            f=open(lumifilename,'rb')
        except IOError:
            print 'failed to open file ',lumifilename
            return result
        freader=csv.reader(f,delimiter=',')
        idx=0
        for row in freader:
            if idx==0:
                idx=1
                continue
            if row[0].find('#')==1:
                continue
            [runnum,fillnum]=map(lambda i:int(i),row[0].split(':'))
            avgpu=float(row[7])
            if avgpu>50: continue
            putime=row[2]
            max_avgpu=avgpu
            max_putime=putime
            if fillnum not in fills:#new fill
                fills.append(fillnum)               
                results[y][0].append(max_putime)
                results[y][1].append(max_avgpu)
            if avgpu>max_avgpu:
                results[y][0][-1]=max_putime
                results[y][1][-1]=max_avgpu
        print results
    return results


######################################################################

def CacheFilePath(cache_file_dir, day=None):
    cache_file_path = os.path.abspath(cache_file_dir)
    if day:
        cache_file_name = "lumicalc_cache_%s.csv" % day.isoformat()
        cache_file_path = os.path.join(cache_file_path, cache_file_name)
    return cache_file_path


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

def TweakPlot(fig, ax, time_range,
              add_extra_head_room=False):

    # Fiddle with axes ranges etc.
    (time_begin, time_end) = time_range
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
    minorXlocator=matplotlib.ticker.AutoMinorLocator()
    #ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    #ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_minor_locator(minorXlocator)

    formatter = matplotlib.dates.DateFormatter(DATE_FMT_STR_AXES)
    ax.xaxis.set_major_formatter(formatter)

    fig.subplots_adjust(top=.85, bottom=.14, left=.13, right=.91)
    # End of TweakPlot().

######################################################################

if __name__ == "__main__":

    desc_str = "test pu" \
               
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
        "beam_fluctuation" : None,
        "verbose" : False,
        "oracle_connection" : None
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

    # WORKAROUND WORKAROUND WORKAROUND
    amodetag_bug_workaround = False
    if accel_mode == "2013_amode_bug_workaround":
        amodetag_bug_workaround = True
        accel_mode = "PAPHYS"
    # WORKAROUND WORKAROUND WORKAROUND end

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

    beam_fluctuation_tmp = cfg_parser.get("general", "beam_fluctuation")
    # If no beam energy fluctuation specified, use the default for
    # this accelerator mode.
    beam_fluctuation = None
    beam_fluctuation_from_cfg = None
    if not beam_fluctuation_tmp:
        print "No beam energy fluctuation specified --> using the defaults to '%s'" % \
              accel_mode
        beam_fluctuation_from_cfg = False
    else:
        beam_fluctuation_from_cfg = True
        beam_fluctuation = float(beam_fluctuation_tmp)

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

    # If an Oracle connection string is specified, use direct Oracle
    # access. Otherwise access passes through the Frontier
    # cache. (Fine, but much slower to receive the data.)
    oracle_connection_string = cfg_parser.get("general", "oracle_connection")
    use_oracle = (len(oracle_connection_string) != 0)

    ##########

    # Map accelerator modes (as fed to lumiCalc) to particle type
    # strings to be used in plot titles etc.
    particle_type_strings = {
        "PROTPHYS" : "pp",
        "IONPHYS" : "PbPb",
        "PAPHYS" : "pPb"
        }
    particle_type_str = particle_type_strings[accel_mode]

    beam_energy_defaults = {
        "PROTPHYS" : {2010 : 3500.,
                      2011 : 3500.,
                      2012 : 4000.,
                      2013 : 1380.1},
        "IONPHYS" : {2010 : 3500.,
                     2011 : 3500.},
        "PAPHYS" : {2013 : 4000.}
        }
    beam_fluctuation_defaults = {
        "PROTPHYS" : {2010 : .15,
                      2011 : .15,
                      2012 : .15,
                      2013 : .15},
        "IONPHYS" : {2010 : .15,
                     2011 : .15},
        "PAPHYS" : {2013 : .15}
        }

    ##########

    # Environment parameter for access to the Oracle DB.
    if use_oracle:
        os.putenv("TNS_ADMIN", "/afs/cern.ch/cms/lumi/DB")

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
        print "Selecting data for default beam energy for '%s' from:" % accel_mode
        for (key, val) in six.iteritems(beam_energy_defaults[accel_mode]):
            print "  %d : %.1f GeV" % (key, val)
    if beam_fluctuation_from_cfg:
        print "Using beam energy fluctuation of +/- %.0f%%" % \
              (100. * beam_fluctuation)
    else:
        print "Using default beam energy fluctuation for '%s' from:" % accel_mode
        for (key, val) in six.iteritems(beam_fluctuation_defaults[accel_mode]):
            print "  %d : +/- %.0f%%" % (key, 100. * val)
    if use_oracle:
        print "Using direct access to the Oracle luminosity database"
    else:
        print "Using access to the luminosity database through the Frontier cache"

    ##########

    # See if the cache file dir exists, otherwise try to create it.
    path_name = CacheFilePath(cache_file_dir)
    if not os.path.exists(path_name):
        if verbose:
            print "Cache file path does not exist: creating it"
        try:
            os.makedirs(path_name)
        except Exception as err:
            print >> sys.stderr, \
                  "ERROR Could not create cache dir: %s" % path_name
            sys.exit(1)

    ##########


    years=[2010,2011,2012]
    
    lumi_data_by_fill_per_year={} #{year:[[timestamp,timestamp],[max_pu,max_pu]]}
    lumi_data_by_fill_per_year=processdata(years)
    
    #lumi_data_by_fill_per_year[2010]=[['05/05/10 05:59:58','06/02/10 10:47:25','07/02/10 12:47:25','08/02/10 11:47:25'],[10.0,2.5,11.3,4.5]]
    #lumi_data_by_fill_per_year[2011]=[['05/05/11 05:59:58','06/02/11 10:47:25','07/02/11 12:47:25','08/02/11 11:47:25','09/05/11 05:59:58','10/02/11 10:47:25','11/02/11 12:47:25'],[20.0,27.4,30.5,40.,22.,15.,45.]]
    #lumi_data_by_fill_per_year[2012]=[['05/05/12 05:59:58','06/02/12 10:47:25','07/02/12 12:47:25','08/02/12 11:47:25','09/05/12 05:59:58','10/02/12 10:47:25','11/02/12 12:47:25','11/02/12 12:47:25'],[10.0,17.4,20.5,30.,32.,25.,33.,42.]]
    
    
    InitMatplotlib()

    ##########

    year_begin = date_begin.isocalendar()[0]
    year_end = date_end.isocalendar()[0]
    # DEBUG DEBUG DEBUG
    assert year_end >= year_begin
    ##########

    # And this is where the plotting starts.
    print "Drawing things..."
    ColorScheme.InitColors()
        
    #----------

    if len(years) > 1:
        print "  peak interactions for %s together" % ", ".join([str(i) for i in years])

        def PlotPeakPUAllYears(lumi_data_by_fill_per_year):
            """Mode 1: years side-by-side"""

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

                    time_begin_ultimate = datetime.datetime.strptime(lumi_data_by_fill_per_year[years[0]][0][0],DATE_FMT_STR_LUMICALC).date() 
                    str_begin_ultimate = time_begin_ultimate.strftime(DATE_FMT_STR_OUT)
                    for (year_index, year) in enumerate(years):

                        lumi_data = lumi_data_by_fill_per_year[year] 
                        times_tmp = [datetime.datetime.strptime(tmp,DATE_FMT_STR_LUMICALC).date() for tmp in lumi_data[0]]
                        times = [matplotlib.dates.date2num(i) for i in times_tmp] #x_axis
                        maxpus = lumi_data[1] # y_axis
                        
                        # NOTE: Special case for 2010.
                        label = None
                        if year == 2010 or year == 2011 :
                            label = r"%d, %s" % \
                                    (year,r'7TeV $\sigma$=71.5mb')
                        else:
                            label = r"%d, %s" % \
                                    (year,r'8TeV $\sigma$=73mb')
                        ax.plot(times,maxpus,
                                color=color_by_year[year],
#                                marker="none", linestyle="solid",
                                marker="o", linestyle='none',
                                linewidth=4,
                                label=label)
                        if is_log:
                            ax.set_yscale("log")
                    
                    time_begin = datetime.datetime(years[0], 1, 1, 0, 0, 0)
                    time_end = datetime.datetime(years[-1], 12, 16, 20, 50,9)
                    str_begin = time_begin.strftime(DATE_FMT_STR_OUT)
                    str_end = time_end.strftime(DATE_FMT_STR_OUT)

                    num_cols = None
                    num_cols = len(years)
                    tmp_x = 0.095
                    tmp_y = .95
                   
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(tmp_x, 0., 1., tmp_y),
                              frameon=False, ncol=num_cols)
                    for t in leg.get_texts():
                        t.set_font_properties(FONT_PROPS_TICK_LABEL)

                    # Set titles and labels.
                    fig.suptitle(r"CMS peak interactions per crossing, %s" % particle_type_str,
                                 fontproperties=FONT_PROPS_SUPTITLE)
                    ax.set_title("Data included from %s to %s UTC \n" % \
                                 (str_begin_ultimate, str_end),
                                 fontproperties=FONT_PROPS_TITLE)
                    ax.set_xlabel(r"Date (UTC)", fontproperties=FONT_PROPS_AX_TITLE)
                    ax.set_ylabel(r"Peak interactions per crossing",\
                                  fontproperties=FONT_PROPS_AX_TITLE)

                    # Add the logo.
                    #zoom = 1.7
                    zoom = .95
                    AddLogo(logo_name, ax, zoom=zoom)
                    extra_head_room = 0
                    if is_log:
                        #if mode == 1:
                        extra_head_room = 1
                        #elif mode == 2:
                        #    extra_head_room = 2
                    TweakPlot(fig, ax, (time_begin, time_end),
#                    TweakPlot(fig, ax, (time_begin_ultimate, time_end),
                              add_extra_head_room=True)

                    log_suffix = ""
                    if is_log:
                        log_suffix = "_log"
                    SavePlot(fig, "peak_pu_%s_%s%s" % \
                             (particle_type_str.lower(),
                              log_suffix, file_suffix))
        PlotPeakPUAllYears(lumi_data_by_fill_per_year)
        
    plt.close()

    print "Done"

######################################################################
