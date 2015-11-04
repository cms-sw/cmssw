import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class ZMuMuValidation(GenericValidationData):
    def __init__(self, valName, alignment, config,
                 configBaseName = "TkAlZMuMuValidation", scriptBaseName = "TkAlZMuMuValidation", crabCfgBaseName = "TkAlZMuMuValidation",
                 resultBaseName = "ZMuMuValidation", outputBaseName = "ZMuMuValidation"):
        defaults = {
            "zmumureference": ("/store/caf/user/emiglior/Alignment/TkAlDiMuonValidation/Reference/BiasCheck_DYToMuMu_Summer12_TkAlZMuMu_IDEAL.root"),
            "resonance": "Z"
            }
        mandatories = ["etamaxneg", "etaminneg", "etamaxpos", "etaminpos"]
        self.configBaseName = configBaseName
        self.scriptBaseName = scriptBaseName
        self.crabCfgBaseName = crabCfgBaseName
        self.resultBaseName = resultBaseName
        self.outputBaseName = outputBaseName
        self.needParentFiles = False
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "zmumu", addDefaults=defaults,
                                       addMandatories=mandatories)
        if self.general["zmumureference"].startswith("/store"):
            self.general["zmumureference"] = "root://eoscms//eos/cms" + self.general["zmumureference"]
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the Z->mumu validation!\n"
                                "Please set parallelJobs = 1.")
    
    def createConfiguration(self, path):
        cfgName = "%s.%s.%s_cfg.py"%( self.configBaseName, self.name,
                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.ZMuMuValidationTemplate}
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path):
        return GenericValidationData.createScript(self, path, template = configTemplates.zMuMuScriptTemplate)

    def createCrabCfg(self, path):
        return GenericValidationData.createCrabCfg(self, path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = GenericValidationData.getRepMap(self, alignment) 
        repMap.update({
            "nEvents": self.general["maxevents"],
            "outputFile": ("0_zmumuHisto.root"
                           ",genSimRecoPlots.root"
                           ",FitParameters.txt"),
            "eosdir": os.path.join(self.general["eosdir"], "%s/%s/%s" % (self.outputBaseName, self.name, alignment.name)),
            "workingdir": ".oO[datadir]Oo./%s/%s/%s" % (self.outputBaseName, self.name, alignment.name),
            "plotsdir": ".oO[datadir]Oo./%s/%s/%s/plots" % (self.outputBaseName, self.name, alignment.name),
                })
        return repMap
