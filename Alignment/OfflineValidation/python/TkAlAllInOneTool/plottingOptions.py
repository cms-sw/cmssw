import os
import random
import globalDictionaries
import configTemplates
from helperFunctions import getCommandOutput2
from TkAlExceptions import AllInOneError


class BasePlottingOptions:
    def __init__(self, config, valType, addDefaults = {}, addMandatories=[]):
        import random
        self.type = valType
        self.general = config.getGeneral()
        self.randomWorkdirPart = "%0i"%random.randint(1,10e9)
        self.config = config

        defaults = {
                    "cmssw" : os.environ["CMSSW_BASE"],
                    "publicationstatus" : "",
                    "customtitle" : "",
                    "era" : "NONE",
                    "legendheader" : "",
                   }
        defaults.update(addDefaults)
        mandatories = []
        mandatories += addMandatories
        theUpdate = config.getResultingSection("plots:"+self.type,
                                               defaultDict = defaults,
                                               demandPars = mandatories)
        self.general.update(theUpdate)



        self.cmssw = self.general["cmssw"]
        badcharacters = r"\'"
        for character in badcharacters:
            if character in self.cmssw:
                raise AllInOneError("The bad characters " + badcharacters + " are not allowed in the cmssw\n"
                                    "path name.  If you really have it in such a ridiculously named location,\n"
                                    "try making a symbolic link somewhere with a decent name.")
        try:
            os.listdir(self.cmssw)
        except OSError:
            raise AllInOneError("Your cmssw release " + self.cmssw + ' does not exist')

        if self.cmssw == os.environ["CMSSW_BASE"]:
            self.scramarch = os.environ["SCRAM_ARCH"]
            self.cmsswreleasebase = os.environ["CMSSW_RELEASE_BASE"]
        else:
            command = ("cd '" + self.cmssw + "' && eval `scramv1 ru -sh 2> /dev/null`"
                       ' && echo "$CMSSW_BASE\n$SCRAM_ARCH\n$CMSSW_RELEASE_BASE"')
            commandoutput = getCommandOutput2(command).split('\n')
            self.cmssw = commandoutput[0]
            self.scramarch = commandoutput[1]
            self.cmsswreleasebase = commandoutput[2]



        self.general["publicationstatus"] = self.general["publicationstatus"].upper()
        self.general["era"] = self.general["era"].upper()

        if not self.general["publicationstatus"] and not self.general["customtitle"]:
            self.general["publicationstatus"] = "INTERNAL"
        if self.general["customtitle"] and not self.general["publicationstatus"]:
            self.general["publicationstatus"] = "CUSTOM"

        if self.general["publicationstatus"] != "CUSTOM" and self.general["customtitle"]:
            raise AllInOneError("If you would like to use a custom title, please leave out the 'publicationstatus' parameter")
        if self.general["publicationstatus"] == "CUSTOM" and not self.general["customtitle"]:
            raise AllInOneError("If you want to use a custom title, you should provide it using 'customtitle' in the [plots:%s] section" % valType)

        publicationstatusenum = ["INTERNAL", "INTERNAL_SIMULATION", "PRELIMINARY", "PUBLIC", "SIMULATION", "UNPUBLISHED", "CUSTOM"]
        eraenum = ["NONE", "CRUZET15", "CRAFT15", "COLL0T15"]
        if self.general["publicationstatus"] not in publicationstatusenum:
            raise AllInOneError("Publication status must be one of " + ", ".join(publicationstatusenum) + "!")
        if self.general["era"] not in eraenum:
            raise AllInOneError("Era must be one of " + ", ".join(eraenum) + "!")

        knownOpts = defaults.keys()+mandatories
        ignoreOpts = []
        config.checkInput("plots:"+self.type,
                          knownSimpleOptions = knownOpts,
                          ignoreOptions = ignoreOpts)

    def getRepMap(self):
        result = self.general
        result.update({
                "workdir": os.path.join(self.general["workdir"],
                                        self.randomWorkdirPart),
                "datadir": self.general["datadir"],
                "logdir": self.general["logdir"],
                "CMSSW_BASE": self.cmssw,
                "SCRAM_ARCH": self.scramarch,
                "CMSSW_RELEASE_BASE": self.cmsswreleasebase,
                })
        return result

class PlottingOptionsTrackSplitting(BasePlottingOptions):
    def __init__(self, config, addDefaults = {}, addMandatories=[]):
        defaults = {}
        defaults.update(addDefaults)
        mandatories = []
        mandatories += addMandatories
        BasePlottingOptions.__init__(self, config, "split", defaults, mandatories)

class PlottingOptionsOffline(BasePlottingOptions):
    def __init__(self, config, addDefaults = {}, addMandatories=[]):
        defaults = {
                    "DMRMethod":"median,rmsNorm",
                    "DMRMinimum":"30",
                    "DMROptions":"",
                    "OfflineTreeBaseDir":"TrackHitFilter",
                    "SurfaceShapes":"none"
                   }
        defaults.update(addDefaults)
        mandatories = []
        mandatories += addMandatories
        BasePlottingOptions.__init__(self, config, "offline", defaults, mandatories)

def PlottingOptions(config, valType):
    plottingOptionsClasses = {
                              "offline": PlottingOptionsOffline,
                              "split": PlottingOptionsTrackSplitting,
                             }

    if valType not in globalDictionaries.plottingOptions:
        globalDictionaries.plottingOptions[valType] = plottingOptionsClasses[valType](config)
    return globalDictionaries.plottingOptions[valType].getRepMap()
