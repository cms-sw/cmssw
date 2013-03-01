import os
import configTemplates
import globalDictionaries
from dataset import Dataset
from genericValidation import GenericValidationMC
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class MonteCarloValidation(GenericValidationMC):
    def __init__(self, valName, alignment, config):
        GenericValidationMC.__init__(self, valName, alignment, config,
                                       "mcValidate")
        mandatories = [ "relvalsample", "maxevents" ]
        mcValidate = config.getResultingSection( "mcValidate:"+self.name, 
                                                 demandPars = mandatories )
        self.general.update( mcValidate )
        if self.general["relvalsample"] not in globalDictionaries.usedDatasets:
            globalDictionaries.usedDatasets[self.general["relvalsample"]] = Dataset(
                self.general["relvalsample"] )
        self.dataset = globalDictionaries.usedDatasets[self.general["relvalsample"]]
        self.general["dataset"] = self.general["relvalsample"]


    def createConfiguration(self, path ):
        cfgName = "TkAlMcValidate.%s.%s_cfg.py"%(self.name,
                                                   self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName:replaceByMap(configTemplates.mcValidateTemplate, repMap)}
        self.filesToCompare[GenericValidationMC.defaultReferenceName] = \
            repMap["outputFile"]
        GenericValidationMC.createConfiguration(self, cfgs, path)

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
        return GenericValidationMC.createScript(self, scripts, path)

    def createCrabCfg(self, path, crabCfgBaseName = "TkAlMcValidate"):
        return GenericValidationMC.createCrabCfg(self, path, crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidationMC.getRepMap(self, alignment)
        repMap.update({
            "outputFile": replaceByMap((".oO[workdir]Oo./McValidation_"
                                        + self.name +
                                        "_.oO[name]Oo..root"), repMap ),
            "nEvents": self.general["maxevents"],
            "RelValSample": self.general["relvalsample"]
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        if self.jobmode.split( ',' )[0] == "crab":
            repMap["outputFile"] = os.path.basename( repMap["outputFile"] )
        return repMap

