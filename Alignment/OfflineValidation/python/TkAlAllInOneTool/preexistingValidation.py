import os
from genericValidation import GenericValidation, GenericValidationData
from geometryComparison import GeometryComparison
from helperFunctions import boolfromstring, getCommandOutput2, parsecolor, parsestyle
from monteCarloValidation import MonteCarloValidation
from offlineValidation import OfflineValidation
from primaryVertexValidation import PrimaryVertexValidation
from plottingOptions import PlottingOptions
from TkAlExceptions import AllInOneError
from trackSplittingValidation import TrackSplittingValidation
from zMuMuValidation import ZMuMuValidation

class PreexistingValidation(GenericValidation):
    """
    Object representing a validation that has already been run,
    but should be included in plots.
    """
    defaults = {"title": ".oO[name]Oo."}
    mandatories = {"file", "color", "style"}
    removemandatories = {"dataset", "maxevents", "trackcollection"}
    def __init__(self, valName, config):
        self.general = config.getGeneral()
        self.name = self.general["name"] = valName
        self.config = config

        theUpdate = config.getResultingSection("preexisting"+self.valType+":"+self.name,
                                               defaultDict = self.defaults,
                                               demandPars = self.mandatories)
        self.general.update(theUpdate)

        self.title = self.general["title"]
        if "|" in self.title or "," in self.title or '"' in self.title:
            msg = "The characters '|', '\"', and ',' cannot be used in the alignment title!"
            raise AllInOneError(msg)
        self.needsproxy = boolfromstring(self.general["needsproxy"], "needsproxy")
        self.jobid = self.general["jobid"]
        if self.jobid:
            try:  #make sure it's actually a valid jobid
                output = getCommandOutput2("bjobs %(jobid)s 2>&1"%self.general)
                if "is not found" in output: raise RuntimeError
            except RuntimeError:
                raise AllInOneError("%s is not a valid jobid.\nMaybe it finished already?"%self.jobid)

        knownOpts = set(self.defaults.keys())|self.mandatories|self.optionals
        ignoreOpts = []
        config.checkInput("preexisting"+self.valType+":"+self.name,
                          knownSimpleOptions = knownOpts,
                          ignoreOptions = ignoreOpts)
        self.jobmode = None

        try:  #initialize plotting options for this validation type
            result = PlottingOptions(self.config, self.valType)
        except KeyError:
            pass

    @property
    def filesToCompare(self):
        return {self.defaultReferenceName: self.general["file"]}

    def getRepMap(self):
        #do not call super
        try:
            result = PlottingOptions(self.config, self.valType)
        except KeyError:
            result = {}
        result.update(self.general)
        result.update({
                       "color": str(parsecolor(result["color"])),
                       "style": str(parsestyle(result["style"])),
                      })
        return result

    def createFiles(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")
    def createConfiguration(self, *args, **kwargs):
        pass
    def createScript(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")
    def createCrabCfg(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingOfflineValidation(PreexistingValidation, OfflineValidation):
    deprecateddefaults = {
            "DMRMethod":"",
            "DMRMinimum":"",
            "DMROptions":"",
            "OfflineTreeBaseDir":"",
            "SurfaceShapes":""
            }
    defaults = deprecateddefaults.copy()
    def __init__(self, valName, config):
        super(PreexistingOfflineValidation, self).__init__(valName, config)
        for option in self.deprecateddefaults:
            if self.general[option]:
                raise AllInOneError("The '%s' option has been moved to the [plots:offline] section.  Please specify it there."%option)

    def getRepMap(self):
        result = super(PreexistingOfflineValidation, self).getRepMap()
        result.update({
                       "filetoplot": self.general["file"],
                     })
        return result

    def appendToMerge(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingPrimaryVertexValidation(PreexistingValidation, PrimaryVertexValidation):
    removemandatories = {"isda","ismc","runboundary","vertexcollection","lumilist","ptCut","etaCut","runControl","numberOfBins"}
    def getRepMap(self):
        result = super(PreexistingPrimaryVertexValidation, self).getRepMap()
        result.update({
                       "filetoplot": self.general["file"],
                     })
        return result

    def appendToMerge(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingTrackSplittingValidation(PreexistingValidation, TrackSplittingValidation):
    def appendToMerge(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingMonteCarloValidation(PreexistingValidation):
    pass

class PreexistingZMuMuValidation(PreexistingValidation):
    def __init__(self, *args, **kwargs):
        raise AllInOneError("Preexisting Z->mumu validation not implemented")
        #more complicated, it has multiple output files

class PreexistingGeometryComparison(PreexistingValidation):
    def __init__(self, *args, **kwargs):
        raise AllInOneError("Preexisting geometry comparison not implemented")
