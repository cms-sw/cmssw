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
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "mcValidate")

    def createConfiguration(self, path ):
        cfgName = "%s.%s.%s_cfg.py"%(self.configBaseName, self.name,
                                     self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.mcValidateTemplate}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["finalResultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path):
        scriptName = "%s.%s.%s.sh"%(self.scriptBaseName, self.name,
                                    self.alignmentToValidate.name)
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"] += \
                repMap["CommandLineTemplate"]%{"cfgFile":cfg, "postProcess":"" }

        scripts = {scriptName: configTemplates.scriptTemplate}
        return GenericValidationData.createScript(self, scripts, path, repMap = repMap)

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

