import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class ZMuMuValidation(GenericValidationData):
    def __init__(self, valName, alignment,config):
        defaults = {
            "zmumureference": ("/store/caf/user/emiglior/Alignment/TkAlDiMuonValidation/Reference/BiasCheck_DYToMuMu_Summer12_TkAlZMuMu_IDEAL.root"),
            "resonance": "Z"
            }
        mandatories = ["etamaxneg", "etaminneg", "etamaxpos", "etaminpos"]
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "zmumu", addDefaults=defaults,
                                       addMandatories=mandatories)
    
    def createConfiguration(self, path, configBaseName = "TkAlZMuMuValidation" ):
        cfgName = "%s.%s.%s_cfg.py"%( configBaseName, self.name,
                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.ZMuMuValidationTemplate}
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path, scriptBaseName = "TkAlZMuMuValidation"):
        scriptName = "%s.%s.%s.sh"%(scriptBaseName, self.name,
                                    self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: configTemplates.zMuMuScriptTemplate}
        return GenericValidationData.createScript(self, scripts, path, repMap = repMap)

    def createCrabCfg(self, path, crabCfgBaseName = "TkAlZMuMuValidation"):
        return GenericValidationData.createCrabCfg(self, path, crabCfgBaseName)

    def getRepMap(self, alignment = None):
        repMap = GenericValidationData.getRepMap(self, alignment) 
        repMap.update({
            "nEvents": self.general["maxevents"],
#             "outputFile": "zmumuHisto.root"
            "outputFile": ("0_zmumuHisto.root"
                           ",genSimRecoPlots.root"
                           ",FitParameters.txt")
                })
        return repMap
