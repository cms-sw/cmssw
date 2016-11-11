import os
from genericValidation import GenericValidation, GenericValidationData
from geometryComparison import GeometryComparison
from helperFunctions import getCommandOutput2, parsecolor, parsestyle
from monteCarloValidation import MonteCarloValidation
from offlineValidation import OfflineValidation
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
    def __init__(self, valName, config, valType):
        self.general = config.getGeneral()
        self.name = self.general["name"] = valName
        self.config = config
        self.filesToCompare = {}

        theUpdate = config.getResultingSection("preexisting"+valType+":"+self.name)
        self.general.update(theUpdate)

        self.title = self.general["title"]
        if "|" in self.title or "," in self.title or '"' in self.title:
            msg = "The characters '|', '\"', and ',' cannot be used in the alignment title!"
            raise AllInOneError(msg)

        self.filesToCompare[self.defaultReferenceName] = \
            self.general["file"]

        knownOpts = self.defaults.keys()+self.mandatories
        ignoreOpts = []
        config.checkInput("preexisting"+valType+":"+self.name,
                          knownSimpleOptions = knownOpts,
                          ignoreOptions = ignoreOpts)
        self.jobmode = None

    def getRepMap(self):
        #do not call super
        result = self.general
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
    defaults = deprecateddefaults
    def __init__(self, valName, config):
        super(PreexistingOfflineValidation, self).__init__(valName, config, "offline")
        for option in self.deprecateddefaults:
            if self.general[option]:
                raise AllInOneError("The '%s' option has been moved to the [plots:offline] section.  Please specify it there."%option)

    def getRepMap(self):
        result = super(PreexistingOfflineValidation, self).getRepMap()
        result.update({
                       "filetoplot": self.getCompareStrings("OfflineValidation", plain=True),
                     })

    def appendToMerge(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingTrackSplittingValidation(PreexistingValidation, TrackSplittingValidation):
    def __init__(self, valName, config):
        super(PreexistingTrackSplittingValidation, self).__init__(valName, config, "split")

    def appendToMerge(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingMonteCarloValidation(PreexistingValidation):
    def __init__(self, valName, config):
        super(PreexistingMonteCarloValidation, self).__init__(valName, config, "mcValidate")

class PreexistingZMuMuValidation(PreexistingValidation):
    def __init__(self, *args, **kwargs):
        raise AllInOneError("Preexisting Z->mumu validation not implemented")
        #more complicated, it has multiple output files

class PreexistingGeometryComparison(PreexistingValidation):
    def __init__(self, *args, **kwargs):
        raise AllInOneError("Preexisting geometry comparison not implemented")
