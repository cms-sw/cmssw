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
    valType = "mcValidate"
    def __init__(self, valName, alignment, config):
        super(MonteCarloValidation, self).__init__(valName, alignment, config)
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the MC validation!\n"
                                "Please set parallelJobs = 1.")

    @property
    def cfgTemplate(self):
        return configTemplates.mcValidateTemplate

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

