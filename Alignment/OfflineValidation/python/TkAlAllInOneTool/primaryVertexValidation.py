import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData_CTSR, ValidationWithPlots, pythonboolstring
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError

class PrimaryVertexValidation(GenericValidationData_CTSR, ValidationWithPlots):
    configBaseName  = "TkAlPrimaryVertexValidation"
    scriptBaseName  = "TkAlPrimaryVertexValidation"
    crabCfgBaseName = "TkAlPrimaryVertexValidation"
    resultBaseName  = "PrimaryVertexValidation"
    outputBaseName  = "PrimaryVertexValidation"
    defaults = {
                "pvvalidationreference": ("/store/caf/user/musich/Alignment/TkAlPrimaryVertexValidation/Reference/PrimaryVertexValidation_test_pvvalidation_upgrade2017_design_mc_48bins.root"),
                "doBPix":"True",
                "doFPix":"True"
               }
    mandatories = {"isda","ismc","runboundary","trackcollection","vertexcollection","lumilist","ptCut","etaCut","runControl","numberOfBins"}
    valType = "primaryvertex"
    def __init__(self, valName, alignment, config):
        super(PrimaryVertexValidation, self).__init__(valName, alignment, config)

        for name in "doBPix", "doFPix":
            self.general[name] = pythonboolstring(self.general[name], name)

        if self.general["pvvalidationreference"].startswith("/store"):
            self.general["pvvalidationreference"] = "root://eoscms//eos/cms" + self.general["pvvalidationreference"]
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the PrimaryVertex validation!\n"
                                "Please set parallelJobs = 1.")

    @property
    def ValidationTemplate(self):
        return configTemplates.PrimaryVertexValidationTemplate

    @property
    def DefinePath(self):
        return configTemplates.PVValidationPath

    @property
    def ValidationSequence(self):
        #never enters anywhere, since we use the custom DefinePath which includes the goodVertexSkim
        return ""

    @property
    def use_d0cut(self):
        return False

    @property
    def ProcessName(self):
        return "PrimaryVertexValidation"

    def createScript(self, path):
        return super(PrimaryVertexValidation, self).createScript(path, template = configTemplates.PVValidationScriptTemplate)

    def createCrabCfg(self, path):
        return super(PrimaryVertexValidation, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = super(PrimaryVertexValidation, self).getRepMap(alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"],
            "VertexCollection": self.general["vertexcollection"],
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
        return (' loadFileList("root://eoscms//eos/cms%(finalResultFile)s",'
                '"PVValidation","%(title)s", %(color)s, %(style)s);\n')%repMap

    @classmethod
    def runPlots(cls, validations):
        return configTemplates.PrimaryVertexPlotExecution

    @classmethod
    def plottingscriptname(cls):
        return "TkAlPrimaryVertexValidationPlot.C"

    @classmethod
    def plottingscripttemplate(cls):
        return configTemplates.PrimaryVertexPlotTemplate

    @classmethod
    def plotsdirname(cls):
        return "PrimaryVertexValidation"
