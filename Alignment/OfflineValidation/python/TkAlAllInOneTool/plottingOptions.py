from __future__ import absolute_import
from builtins import range
import os
import random

from . import globalDictionaries
from . import configTemplates

from .genericValidation import ValidationMetaClass, ValidationWithComparison, ValidationWithPlots
from .helperFunctions import getCommandOutput2, replaceByMap, cppboolstring
from .offlineValidation import OfflineValidation
from .primaryVertexValidation import PrimaryVertexValidation
from .TkAlExceptions import AllInOneError
from .trackSplittingValidation import TrackSplittingValidation
from .zMuMuValidation import ZMuMuValidation
from .overlapValidation import OverlapValidation
from six import with_metaclass

class BasePlottingOptions(with_metaclass(ValidationMetaClass,object)):
    defaults = {
                "cmssw" : os.environ["CMSSW_BASE"],
                "publicationstatus" : "",
                "customtitle" : "",
                "customrighttitle" : "",
                "era" : "NONE",
                "legendheader" : "",
                "legendoptions":"all",
               }
    mandatories = set()
    needpackages = {"Alignment/OfflineValidation"}
    def __init__(self, config, valType):
        import random
        self.type = valType
        self.general = config.getGeneral()
        self.randomWorkdirPart = "%0i"%random.randint(1,10e9)
        self.config = config

        theUpdate = config.getResultingSection("plots:"+self.type,
                                               defaultDict = self.defaults,
                                               demandPars = self.mandatories)
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

        for package in self.needpackages:
            for placetolook in self.cmssw, self.cmsswreleasebase:
                pkgpath = os.path.join(placetolook, "src", package)
                if os.path.exists(pkgpath):
                    self.general[package] = pkgpath
                    break
            else:
                raise AllInOneError("Package {} does not exist in {} or {}!".format(package, self.cmssw, self.cmsswreleasebase))

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

        if self.general["era"] != "NONE" and self.general["customrighttitle"]:
            raise AllInOneError("If you would like to use a custom right title, please leave out the 'era' parameter")

        publicationstatusenum = ["INTERNAL", "INTERNAL_SIMULATION", "PRELIMINARY", "PUBLIC", "SIMULATION", "UNPUBLISHED", "CUSTOM"]
        eraenum = ["NONE", "CRUZET15", "CRAFT15", "COLL0T15"]
        if self.general["publicationstatus"] not in publicationstatusenum:
            raise AllInOneError("Publication status must be one of " + ", ".join(publicationstatusenum) + "!")
        if self.general["era"] not in eraenum:
            raise AllInOneError("Era must be one of " + ", ".join(eraenum) + "!")

        knownOpts = set(self.defaults.keys())|self.mandatories|self.optionals
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
                "validationId": self.validationclass.__name__,
                })
        if issubclass(self.validationclass, ValidationWithPlots):
            result["plottingscriptname"] = self.validationclass.plottingscriptname()
            result["plottingscriptpath"] = ".oO[scriptsdir]Oo./.oO[plottingscriptname]Oo."
            result["PlotsDirName"] = self.validationclass.plotsdirname()
        if issubclass(self.validationclass, ValidationWithComparison):
            result["compareAlignmentsPath"] = self.validationclass.comparealignmentspath()
            result["compareAlignmentsName"] = self.validationclass.comparealignmentsname()
        return result

class PlottingOptionsTrackSplitting(BasePlottingOptions):
    defaults = {
                "outliercut": "-1.0",
                "subdetector": "none",
               }
    needpackages = {"Alignment/CommonAlignmentProducer"}
    validationclass = TrackSplittingValidation
    def __init__(self, config):
        super(PlottingOptionsTrackSplitting, self).__init__(config, "split")
        validsubdets = self.validsubdets()
        if self.general["subdetector"] not in validsubdets:
            raise AllInOneError("'%s' is not a valid subdetector!\n" % self.general["subdetector"] + "The options are: " + ", ".join(validsubdets))

    def validsubdets(self):
        filename = replaceByMap(".oO[Alignment/CommonAlignmentProducer]Oo./python/AlignmentTrackSelector_cfi.py", self.getRepMap())
        with open(filename) as f:
            trackselector = f.read()

        minhitspersubdet = trackselector.split("minHitsPerSubDet")[1].split("(",1)[1]

        parenthesesdepth = 0
        i = 0
        for character in minhitspersubdet:
            if character == "(":
                parenthesesdepth += 1
            if character == ")":
                parenthesesdepth -= 1
            if parenthesesdepth < 0:
                break
            i += 1
        minhitspersubdet = minhitspersubdet[0:i]

        results = minhitspersubdet.split(",")
        empty = []
        for i in range(len(results)):
            results[i] = results[i].split("=")[0].strip().replace("in", "", 1)

        results.append("none")

        return [a for a in results if a]

class PlottingOptionsZMuMu(BasePlottingOptions):
    defaults = {
                "resonance": "Z",
                "switchONfit": "false",
                "rebinphi": "4",
                "rebinetadiff": "2",
                "rebineta": "2",
                "rebinpt": "8",
                "AutoSetRange": "false",                
               }
    needpackages = {"MuonAnalysis/MomentumScaleCalibration"}
    validationclass = ZMuMuValidation
    def __init__(self, config):
        super(PlottingOptionsZMuMu, self).__init__(config, "zmumu")
        self.general["switchONfit"] = cppboolstring(self.general["switchONfit"], "switchONfit")

class PlottingOptionsOffline(BasePlottingOptions):
    defaults = {
                "DMRMethod":"median,rmsNorm",
                "DMRMinimum":"30",
                "DMROptions":"",
                "OfflineTreeBaseDir":"TrackHitFilter",
                "SurfaceShapes":"coarse",
                "bigtext":"false",
                "mergeOfflineParJobsScriptPath": ".oO[scriptsdir]Oo./TkAlOfflineJobsMerge.C",
                "usefit": "false","moduleid": ""
               }
    validationclass = OfflineValidation
    def __init__(self, config):
        super(PlottingOptionsOffline, self).__init__(config, "offline")
        for name in "usefit", "bigtext":
            self.general[name] = cppboolstring(self.general[name], name)


class PlottingOptionsPrimaryVertex(BasePlottingOptions):
    defaults = {
                "autoLimits":"false",
                "doMaps":"false",
                "stdResiduals":"true",
                "m_dxyPhiMax":"40",
                "m_dzPhiMax":"40",
                "m_dxyEtaMax":"40",
                "m_dzEtaMax":"40",
                "m_dxyPhiNormMax":"0.5",
                "m_dzPhiNormMax":"0.5",
                "m_dxyEtaNormMax":"0.5",
                "m_dzEtaNormMax":"0.5",
                "w_dxyPhiMax":"150",
                "w_dzPhiMax":"150",
                "w_dxyEtaMax":"150",
                "w_dzEtaMax":"1000",
                "w_dxyPhiNormMax":"1.8",
                "w_dzPhiNormMax":"1.8",
                "w_dxyEtaNormMax":"1.8",
                "w_dzEtaNormMax":"1.8",
                }
    validationclass = PrimaryVertexValidation
    def __init__(self, config):
        super(PlottingOptionsPrimaryVertex, self).__init__(config, "primaryvertex")
        for name in "autoLimits", "doMaps", "stdResiduals":
            self.general[name] = cppboolstring(self.general[name], name)

class PlottingOptionsOverlap(BasePlottingOptions):
    validationclass = OverlapValidation
    def __init__(self, config):
        super(PlottingOptionsOverlap, self).__init__(config, "overlap")

def PlottingOptions(config, valType):
    plottingOptionsClasses = {
                              "offline": PlottingOptionsOffline,
                              "split": PlottingOptionsTrackSplitting,
                              "zmumu": PlottingOptionsZMuMu,
                              "primaryvertex": PlottingOptionsPrimaryVertex,
                              "overlap": PlottingOptionsOverlap
                             }
    if isinstance(valType, type):
        valType = valType.valType

    if valType not in globalDictionaries.plottingOptions:
        if config is None:
            raise ValueError("Have to provide a config the first time you call PlottingOptions for {}".format(valType))
        globalDictionaries.plottingOptions[valType] = plottingOptionsClasses[valType](config)
    return globalDictionaries.plottingOptions[valType].getRepMap()



