import os
import configTemplates
import globalDictionaries
from dataset import Dataset
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class MonteCarloValidation(GenericValidationData):
    configBaseName = "TkAlMcValidate"
    scriptBaseName = "TkAlMcValidate"
    crabCfgBaseName = "TkAlMcValidate"
    resultBaseName = "McValidation"
    outputBaseName = "McValidation"
    needParentFiles = True
    def __init__(self, valName, alignment, config):
        super(MonteCarloValidation, self).__init__(valName, alignment, config,
                                                   "mcValidate")
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the MC validation!\n"
                                "Please set parallelJobs = 1.")

    def createConfiguration(self, path ):
        cfgName = "%s.%s.%s_cfg.py"%(self.configBaseName, self.name,
                                     self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.mcValidateTemplate}
        self.filesToCompare[self.defaultReferenceName] = \
            repMap["finalResultFile"]
        super(MonteCarloValidation, self).createConfiguration(cfgs, path, repMap = repMap)

    def createScript(self, path):
        return super(MonteCarloValidation, self).createScript(path)

    def createCrabCfg(self, path):
        return super(MonteCarloValidation, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = super(MonteCarloValidation, self).getRepMap(alignment)
        repMap.update({
            "nEvents": self.general["maxevents"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        return repMap

