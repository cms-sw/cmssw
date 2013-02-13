import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class OfflineValidation(GenericValidationData):
    def __init__(self, valName, alignment,config):
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "offline")
        defaults = {
            "DMRMethod":"median",
            "DMRMinimum":"30",
            "DMROptions":"",
            "offlineModuleLevelHistsTransient":"False",
            "offlineModuleLevelProfiles":"False",
            "OfflineTreeBaseDir":"TrackHitFilter",
            "SurfaceShapes":"none"
            }
        mandatories = [ "dataset", "maxevents", "trackcollection" ]
        if not config.has_section( "offline:"+self.name ):
            offline = config.getResultingSection( "general",
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        else:
            offline = config.getResultingSection( "offline:"+self.name, 
                                                  defaultDict = defaults,
                                                  demandPars = mandatories )
        self.general.update( offline )
    
    def createConfiguration(self, path,
                            configBaseName = "TkAlOfflineValidation" ):
        cfgName = "%s.%s.%s_cfg.py"%( configBaseName, self.name,
                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
          
        cfgs = {cfgName:replaceByMap( configTemplates.offlineTemplate, repMap)}
        self.filesToCompare[
            GenericValidationData.defaultReferenceName ] = repMap["resultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path)
        
    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        scriptName = "%s.%s.%s.sh"%( scriptBaseName, self.name,
                                     self.alignmentToValidate.name )
        repMap = GenericValidationData.getRepMap(self)
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate,
                                             repMap ) }
        return GenericValidationData.createScript(self, scripts, path)

    def createCrabCfg( self, path,
                       crabCfgBaseName = "TkAlOfflineValidation"  ):
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
        if self.dataset.dataType() == "mc":
            repMap["McOrData"] = "events = .oO[nEvents]Oo."
        elif self.dataset.dataType() == "data":
            repMap["McOrData"] = "lumis = -1"
            if self.jobmode.split( ',' )[0] == "crab":
                print ("For jobmode 'crab' the parameter 'maxevents' will be "
                       "ignored and all events will be processed.")
        crabCfg = {crabCfgName: replaceByMap( configTemplates.crabCfgTemplate,
                                              repMap ) }
        return GenericValidationData.createCrabCfg( self, crabCfg, path )

    def getRepMap(self, alignment = None):
        repMap = GenericValidationData.getRepMap(self, alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "outputFile": replaceByMap( (".oO[workdir]Oo./AlignmentValidation_"
                                         + self.name +
                                         "_.oO[name]Oo..root"), repMap ),
            "resultFile": replaceByMap( (".oO[datadir]Oo./AlignmentValidation_"
                                         + self.name +
                                         "_.oO[name]Oo..root"), repMap ),
            "TrackSelectionTemplate": configTemplates.TrackSelectionTemplate,
            "LorentzAngleTemplate": configTemplates.LorentzAngleTemplate,
            "offlineValidationMode": "Standalone",
            "offlineValidationFileOutput":
            configTemplates.offlineStandaloneFileOutputTemplate,
            # Keep the following parameters for backward compatibility
            "TrackCollection": self.general["trackcollection"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        repMap["resultFile"] = os.path.abspath( repMap["resultFile"] )
        if self.jobmode.split( ',' )[0] == "crab":
            repMap["outputFile"] = os.path.basename( repMap["outputFile"] )
            repMap["resultFile"] = os.path.basename( repMap["resultFile"] )
        return repMap

    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        if validationsSoFar == "":
            validationsSoFar = ('PlotAlignmentValidation p("%(resultFile)s",'
                                '"%(name)s", %(color)s, %(style)s);\n')%repMap
        else:
            validationsSoFar += ('p.loadFileList("%(resultFile)s", "%(name)s",'
                                 '%(color)s, %(style)s);\n')%repMap
        return validationsSoFar

    def appendToMerge( self, mergesSoFar = "" ):
        """
        append all merges here
        """
        repMap = self.getRepMap()
        mergesSoFar += replaceByMap( configTemplates.mergeOfflineParallelResults, repMap )
        return mergesSoFar

class OfflineValidationParallel(OfflineValidation):
    def __init__(self, valName, alignment,config):
        OfflineValidation.__init__(self, valName, alignment, config)
        defaults = {
            "parallelJobs":"1",
            "jobmode":self.jobmode
            }
        if not config.has_section( "offline:"+self.name ):
            offline = config.getResultingSection( "general",
                                                  defaultDict = defaults )
        else:
            offline = config.getResultingSection( "offline:"+self.name, 
                                                  defaultDict = defaults )
        self.general.update( offline )
        self.__NJobs = self.general["parallelJobs"]

    def createConfiguration(self, path, configBaseName = "TkAlOfflineValidation" ):
        # if offline validation uses N parallel jobs, we create here N cfg files
        numberParallelJobs = int( self.general["parallelJobs"] )
        # limit maximum number of parallel jobs to 40
        # (each output file is approximately 20MB)
        maximumNumberJobs = 40
        if numberParallelJobs > maximumNumberJobs:
            raise AllInOneError, "Maximum allowed number of parallel jobs "+str(maximumNumberJobs)+" exceeded!!!"
        # if maxevents is not specified, cannot calculate number of events for each
        # parallel job, and therefore running only a single job
        if int( self.general["maxevents"] ) == -1:
            raise AllInOneError, "Maximum number of events (maxevents) not specified: cannot use parallel jobs in offline validation"
        if numberParallelJobs > 1:    
            if self.general["offlineModuleLevelHistsTransient"] == "True":
                raise AllInOneError, "To be able to merge results when running parallel jobs, set offlineModuleLevelHistsTransient to false."
        for index in range(numberParallelJobs):
            cfgName = "%s.%s.%s_%s_cfg.py"%( configBaseName, self.name, self.alignmentToValidate.name, str(index) )
            repMap = self.getRepMap()
            # in this parallel job, skip index*(maxEvents/nJobs) events from the beginning
            # (first index is zero, so no skipping for a single job)
            # and use _index_ in the name of the output file
            repMap.update({"nIndex": str(index)})
            # Create the result file directly to datadir since should not use /tmp/
            # see https://cern.service-now.com/service-portal/article.do?n=KB0000484
            repMap.update({
                "outputFile": replaceByMap( ".oO[datadir]Oo./AlignmentValidation_"
                                            + self.name +
                                            "_.oO[name]Oo._.oO[nIndex]Oo..root", repMap )
                })
            repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
            repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )

            cfgs = {cfgName:replaceByMap( configTemplates.offlineParallelTemplate, repMap)}
            self.filesToCompare[ GenericValidationData.defaultReferenceName ] = repMap["resultFile"] 
            GenericValidationData.createConfiguration(self, cfgs, path)
            # here is a small problem. only the last cfgs is saved
            # it requires a bit ugly solution later
        
    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        # A separate script is created for each parallel jobs.
        # Since only one cfg is saved a bit ugly solution is needed in the loop.
        returnValue = []
        numJobs = int( self.general["parallelJobs"] )
        for index in range(numJobs):
            scriptName = "%s.%s.%s_%s.sh"%(scriptBaseName, self.name, self.alignmentToValidate.name, str(index) )
            repMap = GenericValidationData.getRepMap(self)
            repMap["nIndex"]=""
            repMap["nIndex"]=str(index)
            repMap["CommandLine"]=""
            for cfg in self.configFiles:
                # The ugly solution here is to change the name for each parallel job 
                cfgtemp = cfg.replace( str(numJobs-1)+"_cfg.py" , str(index)+"_cfg.py" )
                repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfgtemp,
                                                                       "postProcess":""
                                                                       }
                scripts = {scriptName: replaceByMap( configTemplates.parallelScriptTemplate, repMap ) }
                returnValue.extend( GenericValidationData.createScript(self, scripts, path) )
        return returnValue

    def getRepMap(self, alignment = None):
        repMap = OfflineValidation.getRepMap(self, alignment) 
        repMap.update({
            "nJobs": self.general["parallelJobs"],
            "offlineValidationFileOutput":
            configTemplates.offlineParallelFileOutputTemplate,
            "nameValidation": self.name
            })
        # In case maxevents==-1, set number of parallel jobs to 1
        # since we cannot calculate number of events for each
        # parallel job
        if str(self.general["maxevents"]) == "-1":
            repMap.update({ "nJobs": "1" })
        return repMap

    def appendToMergeParJobs( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned, 
        else the validation is appended to the list
        """
        repMap = self.getRepMap()

        parameters = ""
        fileToAdd = ""
        for index in range(int(self.__NJobs)):
            fileToAdd = '%(resultFile)s'%repMap
            fileToAdd = fileToAdd.replace('.root','_'+str(index)+'.root')
            if index < int( self.general["parallelJobs"] )-1:
                parameters = parameters+fileToAdd+','
            else:
                parameters = parameters+fileToAdd                
                
        mergedoutputfile = "AlignmentValidation_" + self.name + "_" + '%(name)s'%repMap + ".root"
        validationsSoFar += 'hadd("'+parameters+'","'+mergedoutputfile+'");' + "\n"
        return validationsSoFar

    def createCrabCfg( self ):
        raise AllInOneError, ("jobmode 'crab' not supported for "
                              "'offlineParallel' validation. "
                              "Please choose another 'jobmode'.")


class OfflineValidationDQM(OfflineValidation):
    def __init__(self, valName, alignment, config):
        OfflineValidation.__init__(self, valName, alignment, config)
        if not config.has_section("DQM"):
            raise AllInOneError, "You need to have a DQM section in your configfile!"
        
        self.__PrimaryDataset = config.get("DQM", "primaryDataset")
        self.__firstRun = int(config.get("DQM", "firstRun"))
        self.__lastRun = int(config.get("DQM", "lastRun"))

    def createConfiguration(self, path):
        OfflineValidation.createConfiguration(self, path, "TkAlOfflineValidationDQM")
        
    def createScript(self, path):
        return OfflineValidation.createScript(self, path, "TkAlOfflineValidationDQM")

    def getRepMap(self, alignment = None):
        repMap = OfflineValidation.getRepMap(self, alignment)
        repMap.update({
                "workdir": os.path.expandvars(repMap["workdir"]),
		"offlineValidationMode": "Dqm",
                "offlineValidationFileOutput": configTemplates.offlineDqmFileOutputTemplate,
                "workflow": "/%s/TkAl%s-.oO[alignmentName]Oo._R%09i_R%09i_ValSkim-v1/ALCARECO"%(self.__PrimaryDataset, datetime.datetime.now().strftime("%y"), self.__firstRun, self.__lastRun),
                "firstRunNumber": "%i"% self.__firstRun
                }
            )
        if "__" in repMap["workflow"]:
            raise AllInOneError, "the DQM workflow specefication must not contain '__'. it is: %s"%repMap["workflow"]
        return repMap
