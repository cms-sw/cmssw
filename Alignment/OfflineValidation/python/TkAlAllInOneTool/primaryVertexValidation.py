import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError

class PrimaryVertexValidation(GenericValidationData):
    def __init__(self, valName, alignment, config,
                 configBaseName  = "TkAlPrimaryVertexValidation", 
                 scriptBaseName  = "TkAlPrimaryVertexValidation", 
                 crabCfgBaseName = "TkAlPrimaryVertexValidation",
                 resultBaseName  = "PrimaryVertexValidation", 
                 outputBaseName  = "PrimaryVertexValidation"):
        defaults = {
            "pvvalidationreference": ("/store/caf/user/musich/Alignment/TkAlPrimaryVertexValidation/Reference/PrimaryVertexValidation_test_pvvalidation_mc_design_mc_48bins.root"),
            }
        
        mandatories = ["isda","ismc","runboundary","trackcollection","vertexcollection","lumilist","ptCut","runControl","numberOfBins"]
        self.configBaseName  = configBaseName
        self.scriptBaseName  = scriptBaseName
        self.crabCfgBaseName = crabCfgBaseName
        self.resultBaseName  = resultBaseName
        self.outputBaseName  = outputBaseName
        self.needParentFiles = False
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "primaryvertex", addDefaults=defaults,
                                       addMandatories=mandatories)
        
        if self.general["pvvalidationreference"].startswith("/store"):
            self.general["pvvalidationreference"] = "root://eoscms//eos/cms" + self.general["pvvalidationreference"]
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the PrimaryVertex validation!\n"
                                "Please set parallelJobs = 1.")
    
    def createConfiguration(self, path):
        cfgName = "%s.%s.%s_cfg.py"%( self.configBaseName, self.name,
                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.PrimaryVertexValidationTemplate}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["finalResultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path):
        return GenericValidationData.createScript(self, path, template = configTemplates.PVValidationScriptTemplate)

    def createCrabCfg(self, path):
        return GenericValidationData.createCrabCfg(self, path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = GenericValidationData.getRepMap(self, alignment) 
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

    def appendToMerge( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned,
        else the validation is appended to the list
        """
        repMap = self.getRepMap()
        
        parameters = " ".join(os.path.join("root://eoscms//eos/cms", file.lstrip("/")) for file in repMap["resultFiles"])

        mergedoutputfile = os.path.join("root://eoscms//eos/cms", repMap["finalResultFile"].lstrip("/"))
        validationsSoFar += "hadd -f %s %s\n" % (mergedoutputfile, parameters)
        return validationsSoFar

    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()

        if validationsSoFar  != "":
            validationsSoFar += ','
            validationsSoFar += "root://eoscms//eos/cms%(finalResultFile)s=%(title)s"%repMap
        else:
            validationsSoFar += "root://eoscms//eos/cms%(finalResultFile)s=%(title)s"%repMap
        return validationsSoFar
    
