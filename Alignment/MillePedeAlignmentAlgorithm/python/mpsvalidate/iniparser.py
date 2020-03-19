##########################################################################

# Read the ini data which is passed to the function and return the
# data as a configData object. If a parameter is given the function
# parseParameter will override the config values.
##

import configparser as ConfigParser
import logging
import os

from Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper \
    import checked_out_MPS


class ConfigData:
    """ stores the config data of the ini files or the console parameters
    """

    def __init__(self):
        # get path to modules, defaut ini and templates
        self.mpspath = os.path.join(os.environ["CMSSW_BASE"]
                                    if checked_out_MPS()[0]
                                    else os.environ["CMSSW_RELEASE_BASE"], "src",
                                    "Alignment", "MillePedeAlignmentAlgorithm")

        # General
        # jobmX dir
        self.jobNumber = -1
        # MillePedeUser_X time
        self.jobTime = -1
        # ./jobData/jobmX path
        self.jobDataPath = ""
        # base outputpath
        self.outputPath = ""
        # latex file name
        self.latexfile = ""
        # identification in every plot (e.g. mp1885)
        self.message = ""
        # limits for warning dict with keys xyz, rot, dist
        # arguments must be given in this order
        self.limit = {}
        # statboxsize
        self.statboxsize = -1
        # global tag
        self.globalTag = None
        # first run to pick for the geometry in self.globalTag
        self.firstRun = None

        # what should be created
        self.showmonitor    = False
        self.showadditional = False
        self.showdump       = False
        self.showtime       = False
        self.showhighlevel  = False
        self.showmodule     = False
        self.showsubmodule  = False
        self.showtex        = False
        self.showbeamer     = False
        self.showhtml       = False

        # MODULEPLOTS
        # number of bins after shrinking
        self.numberofbins = -1
        # definition of sharp peak; max_outlier / StdDev > X
        self.defpeak = -1
        # new histogram width in units of StdDev
        self.widthstddev = -1
        # every parameter (e.g. xyz) with same range
        self.samerange = False
        # rangemode "stddev" = multiple of StdDev, "all" = show all, "given" =
        # use given ranges
        self.rangemode = -1
        # ranges
        self.rangexyzM = []
        self.rangerotM = []
        self.rangedistM = []

        # HIGHLEVELPLOTS
        # given ranges
        self.rangexyzHL = []
        self.rangerotHL = []
        # every parameter (e.g. xyz) with same range
        self.samerangeHL = False
        # rangemode "all" = show all, "given" = use given ranges
        self.rangemodeHL = -1

        # Time dependent
        self.firsttree = -1

        # list with the plots for the output
        self.outputList = []

    def parseConfig(self, path):
        logger = logging.getLogger("mpsvalidate")

        # create ConfigParser object
        parser = ConfigParser.ConfigParser()

        # read ini file
        if (parser.read(path) == []):
            logger.error("Could not open ini-file: {0}".format(path))

        # collect data and process it
        try:
            self.jobNumber = parser.getint("GENERAL", "job")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        try:
            self.jobDataPath = parser.get("GENERAL", "jobdatapath")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        # if jobData path is given
        if self.jobDataPath != "":
            self.outputPath = "validation_output"

        # set jobDataPath if job number is given and if path is not given
        if self.jobNumber != -1 and self.jobDataPath == "":
            if self.jobNumber == 0:
                self.jobDataPath = "jobData/jobm"
            else:
                self.jobDataPath = "jobData/jobm{0}".format(self.jobNumber)
            self.outputPath = os.path.join(self.jobDataPath, "validation_output")

        # set outputpath
        try:
            if parser.get("GENERAL", "outputpath"):
                self.outputPath = parser.get("GENERAL", "outputpath")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass


        # data which could be stored directly
        try:
            self.jobTime = parser.getint("GENERAL", "time")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        try:
            self.latexfile = parser.get("GENERAL", "latexfile")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        try:
            self.limit = parser.get("GENERAL", "limit")
            self.limit = map(float, "".join(self.limit.split()).split(","))
            # make a dict to lookup by mode
            self.limit = dict(zip(["xyz", "rot", "dist"], self.limit))
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        try:
            self.statboxsize = parser.getfloat("GENERAL", "statboxsize")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass


        # MODULEPLOTS

        # integers
        for par in ("numberofbins", "defpeak", "widthstddev"):
            try:
                setattr(self, par, parser.getint("MODULEPLOTS", par))
            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                pass

        # booleans
        try:
            self.samerange = parser.getboolean("MODULEPLOTS", "samerange")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        # strings
        try:
            self.rangemode = parser.get("MODULEPLOTS", "rangemode")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        # ranges
        for r in ("rangexyz", "rangerot", "rangedist"):
            try:
                setattr(self, r+"M", parser.get("MODULEPLOTS", r))
                setattr(self, r+"M",
                        sorted(map(float, "".join(getattr(self, r+"M").split()).split(","))))
            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                pass


        # HIGHLEVELPLOTS

        # booleans
        try:
            self.samerangeHL = parser.getboolean("HIGHLEVELPLOTS", "samerange")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        # strings
        try:
            self.rangemodeHL = parser.get("HIGHLEVELPLOTS", "rangemode")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        # ranges
        for r in ("rangexyz", "rangerot"):
            try:
                setattr(self, r+"HL", parser.get("HIGHLEVELPLOTS", r))
                setattr(self, r+"HL",
                        sorted(map(float, "".join(getattr(self, r+"HL").split()).split(","))))
            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                pass


        # TIMEPLOTS

        try:
            self.firsttree = parser.getint("TIMEPLOTS", "firsttree")
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
            pass

        # SHOW
        for boolean in ("showmonitor", "showadditional", "showdump", "showtime",
                        "showhighlevel", "showmodule", "showsubmodule",
                        "showtex", "showbeamer", "showhtml"):
            try:
                setattr(self, boolean, parser.getboolean("SHOW", boolean))
            except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
                pass


    def parseParameter(self, args):
        logger = logging.getLogger("mpsvalidate")

        # check if parameter is given and override the config data
        if args.time != -1:
            self.jobTime = args.time

        if args.job != -1:
            self.jobNumber = args.job

            # set jobDataPath
            if self.jobNumber == 0:
                self.jobDataPath = "jobData/jobm"
            else:
                self.jobDataPath = "jobData/jobm{0}".format(self.jobNumber)
            self.outputPath = os.path.join(self.jobDataPath, "validation_output")

        if args.jobdatapath != "":
            self.jobDataPath = args.jobdatapath

        if args.message != "":
            self.message = args.message

        # if path is given put the output in the current directory
        if args.jobdatapath:
            self.outputPath = "validation_output"

        if args.outputpath:
            self.outputPath = args.outputpath

