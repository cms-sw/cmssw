import os
from genericValidation import GenericValidation, GenericValidationData
from offlineValidation import OfflineValidation
from trackSplittingValidation import TrackSplittingValidation
from monteCarloValidation import MonteCarloValidation
from zMuMuValidation import ZMuMuValidation
from geometryComparison import GeometryComparison
from TkAlExceptions import AllInOneError

class PreexistingValidation(GenericValidation):
    """
    Object representing a validation that has already been run,
    but should be included in plots.
    """
    def __init__(self, valName, config, valType,
                 addDefaults = {}, addMandatories=[]):
        self.name = valName
        self.general = config.getGeneral()
        self.config = config
        self.filesToCompare = {}

        defaults = {"title": self.name}
        defaults.update(addDefaults)
        mandatories = ["file", "color", "style"]
        mandatories += addMandatories

        theUpdate = config.getResultingSection("preexisting"+valType+":"+self.name,
                                               defaultDict = defaults,
                                               demandPars = mandatories)
        self.general.update(theUpdate)

        self.title = self.general["title"]
        if "|" in self.title or "," in self.title or '"' in self.title:
            msg = "The characters '|', '\"', and ',' cannot be used in the alignment title!"
            raise AllInOneError(msg)

        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            self.general["file"]

        knownOpts = defaults.keys()+mandatories
        ignoreOpts = []
        config.checkInput("preexisting"+valType+":"+self.name,
                          knownSimpleOptions = knownOpts,
                          ignoreOptions = ignoreOpts)

    def getRepMap(self):
        result = self.general
        return result

    def getCompareStrings( self, requestId = None, plain = False ):
        result = {}
        repMap = self.getRepMap()
        for validationId in self.filesToCompare:
            repMap["file"] = self.filesToCompare[ validationId ]
            if repMap["file"].startswith( "/castor/" ):
                repMap["file"] = "rfio:%(file)s"%repMap
            elif repMap["file"].startswith( "/store/" ):
                repMap["file"] = "root://eoscms.cern.ch//eos/cms%(file)s"%repMap
            if plain:
                result[validationId]=repMap["file"]
            else:
                result[validationId]= "%(file)s=%(title)s|%(color)s|%(style)s"%repMap
        if requestId == None:
            return result
        else:
            if not "." in requestId:
                requestId += ".%s"%GenericValidation.defaultReferenceName
            if not requestId.split(".")[-1] in result:
                msg = ("could not find %s in reference Objects!"
                       %requestId.split(".")[-1])
                raise AllInOneError(msg)
            return result[ requestId.split(".")[-1] ]

    def createFiles(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")
    def createConfiguration(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")
    def createScript(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")
    def createCrabCfg(self, *args, **kwargs):
        raise AllInOneError("Shouldn't be here...")

class PreexistingOfflineValidation(PreexistingValidation):
    def __init__(self, valName, config,
                 addDefaults = {}, addMandatories=[]):
        defaults = {
            "DMRMethod":"median,rmsNorm",
            "DMRMinimum":"30",
            "DMROptions":"",
            "OfflineTreeBaseDir":"TrackHitFilter",
            "SurfaceShapes":"none"
            }
        defaults.update(addDefaults)
        PreexistingValidation.__init__(self, valName, config, "offline",
                                       defaults, addMandatories)
    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        repMap["file"] = self.getCompareStrings("OfflineValidation", plain = True)
        if validationsSoFar == "":
            validationsSoFar = ('PlotAlignmentValidation p("%(file)s",'
                                '"%(title)s", %(color)s, %(style)s);\n')%repMap
        else:
            validationsSoFar += ('  p.loadFileList("%(file)s", "%(title)s",'
                                 '%(color)s, %(style)s);\n')%repMap
        return validationsSoFar

class PreexistingTrackSplittingValidation(PreexistingValidation):
    def __init__(self, valName, config,
                 addDefaults = {}, addMandatories=[]):
        PreexistingValidation.__init__(self, valName, config, "split",
                                       addDefaults, addMandatories)
    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        comparestring = self.getCompareStrings("TrackSplittingValidation")
        if validationsSoFar != "":
            validationsSoFar += ',"\n              "'
        validationsSoFar += comparestring
        return validationsSoFar

class PreexistingMonteCarloValidation(PreexistingValidation):
    def __init__(self, valName, config,
                 addDefaults = {}, addMandatories=[]):
        PreexistingValidation.__init__(self, valName, config, "mcValidate",
                                       addDefaults, addMandatories)

class PreexistingZMuMuValidation(PreexistingValidation):
    def __init__(self, *args, **kwargs):
        raise AllInOneError("Preexisting Z->mumu validation not implemented")
        #more complicated, it has multiple output files

class PreexistingGeometryComparison(PreexistingValidation):
    def __init__(self, *args, **kwargs):
        raise AllInOneError("Preexisting geometry comparison not implemented")
