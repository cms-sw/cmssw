import os
import configTemplates
import globalDictionaries
from dataset import Dataset
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class MonteCarloValidation(GenericValidationData):
    def __init__(self, valName, alignment, config,
                 configBaseName = "TkAlMcValidate", scriptBaseName = "TkAlMcValidate", crabCfgBaseName = "TkAlMcValidate",
                 resultBaseName = "McValidation", outputBaseName = "McValidation"):
        self.configBaseName = configBaseName
        self.scriptBaseName = scriptBaseName
        self.crabCfgBaseName = crabCfgBaseName
        self.resultBaseName = resultBaseName
        self.outputBaseName = outputBaseName
        self.needParentFiles = True
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "mcValidate")
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the MC validation!\n"
                                "Please set parallelJobs = 1.")

    def createConfiguration(self, path ):
        cfgName = "%s.%s.%s_cfg.py"%(self.configBaseName, self.name,
                                     self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.mcValidateTemplate}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["finalResultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path):
        return GenericValidationData.createScript(self, path)

    def createCrabCfg(self, path):
        return GenericValidationData.createCrabCfg(self, path, self.crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidationData.getRepMap(self, alignment)
        repMap.update({
            "nEvents": self.general["maxevents"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        return repMap

