import os
import configTemplates
import globalDictionaries
from dataset import Dataset
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class MonteCarloValidation(GenericValidationData):
    def __init__(self, valName, alignment, config):
        mandatories = [ "dataset", "maxevents" ]
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "mcValidate", addMandatories=mandatories)

    def createConfiguration(self, path ):
        cfgName = "TkAlMcValidate.%s.%s_cfg.py"%(self.name,
                                                 self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName:replaceByMap(configTemplates.mcValidateTemplate,
                                     repMap)}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["resultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlMcValidate.%s.%s.sh"%(self.name,
                                                self.alignmentToValidate.name)
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"] += \
                repMap["CommandLineTemplate"]%{"cfgFile":cfg, "postProcess":"" }

        scripts = {scriptName: replaceByMap(configTemplates.scriptTemplate,
                                            repMap)}
        return GenericValidationData.createScript(self, scripts, path)

    def createCrabCfg(self, path, crabCfgBaseName = "TkAlMcValidate"):
        return GenericValidationData.createCrabCfg(self, path,
                                                   crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidationData.getRepMap(self, alignment)
        repMap.update({
            "resultFile": replaceByMap(("/store/caf/user/$USER/.oO[eosdir]Oo."
                                        "/McValidation_"
                                        + self.name +
                                        "_.oO[name]Oo..root"), repMap ),
            "outputFile": replaceByMap(("McValidation_"
                                        + self.name +
                                        "_.oO[name]Oo..root"), repMap ),
            "nEvents": self.general["maxevents"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        return repMap

