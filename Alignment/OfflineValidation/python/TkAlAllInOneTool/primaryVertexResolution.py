import os
from . import configTemplates
from . import globalDictionaries
from .genericValidation import GenericValidationData, ValidationWithPlots, pythonboolstring
from .helperFunctions import replaceByMap
from .TkAlExceptions import AllInOneError

class PrimaryVertexResolution(GenericValidationData, ValidationWithPlots):
    configBaseName  = "TkAlPrimaryVertexResolution"
    scriptBaseName  = "TkAlPrimaryVertexResolution"
    crabCfgBaseName = "TkAlPrimaryVertexResolution"
    resultBaseName  = "PrimaryVertexResolution"
    outputBaseName  = "PrimaryVertexResolution"
    defaults = {
        # N.B.: the reference needs to be updated each time the format of the output is changed
        "pvresolutionreference": ("/store/group/alca_trackeralign/validation/PVResolution/Reference/PrimaryVertexResolution_phaseIMC92X_upgrade2017_Ideal.root"),
        "multiIOV":"False",
    }
    #mandatories = {"isda","ismc","runboundary","trackcollection","vertexcollection","lumilist","ptCut","etaCut","runControl","numberOfBins"}
    mandatories = {"runControl","runboundary","doTriggerSelection","triggerBits","trackcollection"}
    valType = "pvresolution"
    def __init__(self, valName, alignment, config):
        super(PrimaryVertexResolution, self).__init__(valName, alignment, config)

        if self.general["pvresolutionreference"].startswith("/store"):
            self.general["pvresolutionreference"] = "root://eoscms//eos/cms" + self.general["pvresolutionreference"]
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the SplotVertexResolution validation!\n"
                                "Please set parallelJobs = 1.")
    @property
    def ValidationTemplate(self):
        return configTemplates.PrimaryVertexResolutionTemplate

    @property
    def TrackSelectionRefitting(self):
        return configTemplates.SingleTrackRefitter

    @property
    def DefinePath(self):
        return configTemplates.PVResolutionPath

    @property
    def ValidationSequence(self):
        #never enters anywhere, since we use the custom DefinePath which includes the goodVertexSkim
        return ""

    @property
    def ProcessName(self):
        return "PrimaryVertexResolution"

    def createScript(self, path):
        return super(PrimaryVertexResolution, self).createScript(path, template = configTemplates.PVResolutionScriptTemplate)

    def createCrabCfg(self, path):
        return super(PrimaryVertexResolution, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = super(PrimaryVertexResolution, self).getRepMap(alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"],
            "eosdir": os.path.join(self.general["eosdir"]),
            #"eosdir": os.path.join(self.general["eosdir"], "%s/%s/%s" % (self.outputBaseName, self.name, alignment.name)),
            "workingdir": ".oO[datadir]Oo./%s/%s/%s" % (self.outputBaseName, self.name, alignment.name),
            "plotsdir": ".oO[datadir]Oo./%s/%s/%s/plots" % (self.outputBaseName, self.name, alignment.name),
            })

        return repMap

    def appendToMerge(self):
        """
        if no argument or "" is passed a string with an instantiation is returned,
        else the validation is appended to the list
        """
        repMap = self.getRepMap()

        parameters = " ".join(os.path.join("root://eoscms//eos/cms", file.lstrip("/")) for file in repMap["resultFiles"])

        mergedoutputfile = os.path.join("root://eoscms//eos/cms", repMap["finalResultFile"].lstrip("/"))
        return "hadd -f %s %s\n" % (mergedoutputfile, parameters)

    def appendToPlots(self):
        repMap = self.getRepMap()
        return (' PVResolution::loadFileList("root://eoscms//eos/cms%(finalResultFile)s",'
                '"PrimaryVertexResolution","%(title)s", %(color)s, %(style)s);\n')%repMap

    @classmethod
    def runPlots(cls, validations):
        return configTemplates.PVResolutionPlotExecution

    @classmethod
    def plottingscriptname(cls):
        return "TkAlPrimaryVertexResolutionPlot.C"

    @classmethod
    def plottingscripttemplate(cls):
        return configTemplates.PVResolutionPlotTemplate

    @classmethod
    def plotsdirname(cls):
        return "PrimaryVertexResolution"
