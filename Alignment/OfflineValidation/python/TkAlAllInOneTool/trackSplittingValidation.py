import os
import configTemplates
from genericValidation import GenericValidation
from helperFunctions import replaceByMap

class TrackSplittingValidation(GenericValidation):
    def __init__(self, valName, alignment, config):
        GenericValidation.__init__(self, valName, alignment, config)
        defaults = {
            "jobmode":self.jobmode,
            "runRange":"",
            "firstRun":"",
            "lastRun":"",
            "begin":"",
            "end":"",
            "JSON":""
            }
        mandatories = [ "trackcollection", "maxevents" ]
        if not config.has_section( "split:"+self.name ):
            split = config.getResultingSection( "general",
                                                defaultDict = defaults,
                                                demandPars = mandatories )
        else:
            split = config.getResultingSection( "split:"+self.name, 
                                                defaultDict = defaults,
                                                demandPars = mandatories )
        self.general.update( split )
        self.jobmode = self.general["jobmode"]


    def createConfiguration(self, path ):
        cfgName = "TkAlTrackSplitting.%s.%s_cfg.py"%( self.name,
                                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap.update({
                "outputFile": replaceByMap( (".oO[workdir]Oo./TrackSplitting_"
                                             + self.name +
                                             "_.oO[name]Oo..root"),
                                            repMap )
                })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        cfgs = {cfgName:replaceByMap( configTemplates.TrackSplittingTemplate, repMap)}
        self.filesToCompare[ GenericValidation.defaultReferenceName ] = repMap["outputFile"]
        GenericValidation.createConfiguration(self, cfgs, path)

    def createScript(self, path):
        scriptName = "TkAlTrackSplitting.%s.%s.sh"%( self.name,
                                                     self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }

        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate, repMap ) }
        return GenericValidation.createScript(self, scripts, path)

    def createCrabCfg( self, path,
                       crabCfgBaseName = "TkAlTrackSplitting"  ):
        crabCfgName = "crab.%s.%s.%s.cfg"%( crabCfgBaseName, self.name,
                                            self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["script"] = "dummy_script.sh"
        repMap["crabOutputDir"] = os.path.basename( path )
        repMap["crabWorkingDir"] = crabCfgName.split( '.cfg' )[0]
        self.crabWorkingDir = repMap["crabWorkingDir"]
        repMap["numberOfJobs"] = self.general["parallelJobs"]
        repMap["cfgFile"] = self.configFiles[0]
        repMap["queue"] = self.jobmode.split( ',' )[1].split( '-q' )[1]
        crabCfg = {crabCfgName: replaceByMap( configTemplates.crabCfgTemplate,
                                              repMap ) }
        return GenericValidation.createCrabCfg( self, crabCfg, path )

    def getRepMap( self, alignment = None ):
        repMap = GenericValidation.getRepMap(self)
        # repMap = self.getRepMap()
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            # Keep the following parameters for backward compatibility
            "TrackCollection": self.general["trackcollection"]
            })
        return repMap
