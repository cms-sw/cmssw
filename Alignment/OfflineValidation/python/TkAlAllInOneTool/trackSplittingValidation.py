import os
import configTemplates
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class TrackSplittingValidation(GenericValidationData):
    def __init__(self, valName, alignment, config):
        mandatories = ["trackcollection", "maxevents", "dataset"]
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "split", addMandatories = mandatories)

    def createConfiguration(self, path ):
        cfgName = "TkAlTrackSplitting.%s.%s_cfg.py"%(self.name,
                                                     self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName:replaceByMap(configTemplates.TrackSplittingTemplate,
                                     repMap)}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["resultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlTrackSplitting.%s.%s.sh"%(self.name,
                                                    self.alignmentToValidate.name)
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= (repMap["CommandLineTemplate"]
                                     %{"cfgFile":cfg, "postProcess":""})

        scripts = {scriptName: replaceByMap(configTemplates.scriptTemplate,
                                            repMap)}
        return GenericValidationData.createScript(self, scripts, path)

    def createCrabCfg(self, path, crabCfgBaseName = "TkAlTrackSplitting"):
        return GenericValidationData.createCrabCfg(self, path, crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidationData.getRepMap(self)
        repMap.update({ 
            "resultFile": replaceByMap( ("/store/caf/user/$USER/.oO[eosdir]Oo."
                                         "/TrackSplitting_"
                                         + self.name +
                                         "_.oO[name]Oo..root"),
                                        repMap ),
            "outputFile": replaceByMap( ("TrackSplitting_"
                                         + self.name +
                                         "_.oO[name]Oo..root"),
                                        repMap ),
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        # repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        # if self.jobmode.split( ',' )[0] == "crab":
        #     repMap["outputFile"] = os.path.basename( repMap["outputFile"] )
        return repMap
