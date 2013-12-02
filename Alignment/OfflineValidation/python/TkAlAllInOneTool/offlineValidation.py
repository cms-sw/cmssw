import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class OfflineValidation(GenericValidationData):
    def __init__(self, valName, alignment,config):
        defaults = {
            "DMRMethod":"median,rmsNorm",
            "DMRMinimum":"30",
            "DMROptions":"",
            "offlineModuleLevelHistsTransient":"False",
            "offlineModuleLevelProfiles":"False",
            "OfflineTreeBaseDir":"TrackHitFilter",
            "SurfaceShapes":"none"
            }
        mandatories = [ "dataset", "maxevents", "trackcollection" ]
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "offline", addDefaults=defaults,
                                       addMandatories=mandatories)
    
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
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfg,
                                                  "postProcess":""
                                                  }
        scripts = {scriptName: replaceByMap( configTemplates.scriptTemplate,
                                             repMap ) }
        return GenericValidationData.createScript(self, scripts, path)

    def createCrabCfg(self, path, crabCfgBaseName = "TkAlOfflineValidation"):
        return GenericValidationData.createCrabCfg(self, path, crabCfgBaseName)

    def getRepMap(self, alignment = None):
        repMap = GenericValidationData.getRepMap(self, alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "outputFile": replaceByMap( ("AlignmentValidation_"
                                         + self.name +
                                         "_.oO[name]Oo..root"), repMap ),
            "resultFile": replaceByMap( ("/store/caf/user/$USER/.oO[eosdir]Oo."
                                         "/AlignmentValidation_"
                                         + self.name +
                                         "_.oO[name]Oo..root"), repMap ),
            "TrackSelectionTemplate": configTemplates.TrackSelectionTemplate,
            "LorentzAngleTemplate": configTemplates.LorentzAngleTemplate,
            "offlineValidationMode": "Standalone",
            "offlineValidationFileOutput":
            configTemplates.offlineStandaloneFileOutputTemplate,
            "TrackCollection": self.general["trackcollection"]
            })
        repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )
        repMap["resultFile"] = os.path.expandvars( repMap["resultFile"] )
        return repMap

    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        if validationsSoFar == "":
            validationsSoFar = ('PlotAlignmentValidation p("%(outputFile)s",'
                                '"%(name)s", %(color)s, %(style)s);\n')%repMap
        else:
            validationsSoFar += ('p.loadFileList("%(outputFile)s", "%(name)s",'
                                 '%(color)s, %(style)s);\n')%repMap
        return validationsSoFar

    def appendToMerge( self, mergesSoFar = "" ):
        """
        append all merges here
        """
        repMap = self.getRepMap()
        mergesSoFar += replaceByMap(configTemplates.mergeOfflineParallelResults,
                                    repMap)
        return mergesSoFar


class OfflineValidationParallel(OfflineValidation):
    def __init__(self, valName, alignment,config):
        OfflineValidation.__init__(self, valName, alignment, config)
        defaults = {
            "parallelJobs":"1",
            "jobmode":self.jobmode
            }
        offline = config.getResultingSection( "offline:"+self.name, 
                                              defaultDict = defaults )
        self.general.update( offline )
        self.__NJobs = self.general["parallelJobs"]
        self.outputFiles = []
        for index in range(int(self.general["parallelJobs"])):
            fName = replaceByMap("AlignmentValidation_"+self.name
                                 +"_.oO[name]Oo._%d.root"%(index),
                                 self.getRepMap())
            self.outputFiles.append(fName)
        

    def createConfiguration(self, path, configBaseName = "TkAlOfflineValidation"):
        # if offline validation uses N parallel jobs, we create here N cfg files
        numberParallelJobs = int( self.general["parallelJobs"] )
        # limit maximum number of parallel jobs to 40
        # (each output file is approximately 20MB)
        maximumNumberJobs = 40
        if numberParallelJobs > maximumNumberJobs:
            msg = ("Maximum allowed number of parallel jobs "
                   +str(maximumNumberJobs)+" exceeded!!!")
            raise AllInOneError(msg)
        # if maxevents is not specified, cannot calculate number of events for
        # each parallel job, and therefore running only a single job
        if int( self.general["maxevents"] ) == -1:
            msg = ("Maximum number of events (maxevents) not specified: "
                   "cannot use parallel jobs in offline validation")
            raise AllInOneError(msg)
        if numberParallelJobs > 1:    
            if self.general["offlineModuleLevelHistsTransient"] == "True":
                msg = ("To be able to merge results when running parallel jobs,"
                       " set offlineModuleLevelHistsTransient to false.")
                raise AllInOneError(msg)
        for index in range(numberParallelJobs):
            cfgName = "%s.%s.%s_%s_cfg.py"%(configBaseName, self.name,
                                            self.alignmentToValidate.name,
                                            str(index))
            repMap = self.getRepMap()
            # in this parallel job, skip index*(maxEvents/nJobs) events from
            # the beginning
            # (first index is zero, so no skipping for a single job)
            # and use _index_ in the name of the output file
            repMap.update({"nIndex": str(index)})
            # Create the result file directly to datadir since should not use /tmp/
            # see https://cern.service-now.com/service-portal/article.do?n=KB0000484
            repMap.update({"outputFile": self.outputFiles[index]})
            repMap["outputFile"] = os.path.expandvars( repMap["outputFile"] )

            cfgs = {cfgName:replaceByMap(configTemplates.offlineParallelTemplate,
                                         repMap)}
            self.filesToCompare[GenericValidationData.defaultReferenceName] = repMap["resultFile"] 
            GenericValidationData.createConfiguration(self, cfgs, path)
            # here is a small problem. only the last cfgs is saved
            # it requires a bit ugly solution later
        
    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        # A separate script is created for each parallel jobs.
        # Since only one cfg is saved a bit ugly solution is needed in the loop.
        returnValue = []
        numJobs = int( self.general["parallelJobs"] )
        for index in range(numJobs):
            scriptName = "%s.%s.%s_%s.sh"%(scriptBaseName, self.name, 
                                           self.alignmentToValidate.name,
                                           str(index))
            repMap = self.getRepMap()
            repMap["nIndex"]=""
            repMap["nIndex"]=str(index)
            repMap["CommandLine"]=""
            repMap.update({"outputFile": self.outputFiles[index]})
            for cfg in self.configFiles:
                # The ugly solution here is to change the name for each parallel job 
                cfgtemp = cfg.replace(str(numJobs-1)+"_cfg.py",
                                      str(index)+"_cfg.py")
                repMap["CommandLine"]+= repMap["CommandLineTemplate"]%{"cfgFile":cfgtemp,
                                                                       "postProcess":""
                                                                       }
                scripts = {scriptName: replaceByMap(configTemplates.parallelScriptTemplate,
                                                    repMap ) }
                returnValue.extend(GenericValidationData.createScript(self,
                                                                      scripts,
                                                                      path) )
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
            fileToAdd = '%(outputFile)s'%repMap
            fileToAdd = fileToAdd.replace('.root','_'+str(index)+'.root')
            if index < int( self.general["parallelJobs"] )-1:
                parameters = parameters+fileToAdd+','
            else:
                parameters = parameters+fileToAdd                
                
        mergedoutputfile = ("AlignmentValidation_" + self.name + "_"
                            + '%(name)s'%repMap + ".root")
        validationsSoFar += ('root -x -b -q "TkAlOfflineJobsMerge.C(\\\"'
                             +parameters+'\\\",\\\"'+mergedoutputfile+'\\\")"'
                             +"\n")
        return validationsSoFar

    def createCrabCfg( self ):
        msg =  ("jobmode 'crab' not supported for 'offlineParallel' validation."
                " Please choose another 'jobmode'.")
        raise AllInOneError(msg)


class OfflineValidationDQM(OfflineValidation):
    def __init__(self, valName, alignment, config):
        OfflineValidation.__init__(self, valName, alignment, config)
        if not config.has_section("DQM"):
            msg = "You need to have a DQM section in your configfile!"
            raise AllInOneError(msg)
        
        self.__PrimaryDataset = config.get("DQM", "primaryDataset")
        self.__firstRun = int(config.get("DQM", "firstRun"))
        self.__lastRun = int(config.get("DQM", "lastRun"))

    def createConfiguration(self, path):
        OfflineValidation.createConfiguration(self, path,
                                              "TkAlOfflineValidationDQM")
        
    def createScript(self, path):
        return OfflineValidation.createScript(self, path,
                                              "TkAlOfflineValidationDQM")

    def getRepMap(self, alignment = None):
        repMap = OfflineValidation.getRepMap(self, alignment)
        repMap.update({
                "workdir": os.path.expandvars(repMap["workdir"]),
		"offlineValidationMode": "Dqm",
                "offlineValidationFileOutput": configTemplates.offlineDqmFileOutputTemplate,
                "workflow": ("/%s/TkAl%s-.oO[alignmentName]Oo._R%09i_R%09i_"
                             "ValSkim-v1/ALCARECO"
                             %(self.__PrimaryDataset,
                               datetime.datetime.now().strftime("%y"),
                               self.__firstRun, self.__lastRun)),
                "firstRunNumber": "%i"% self.__firstRun
                })
        if "__" in repMap["workflow"]:
            msg = ("the DQM workflow specefication must not contain '__'. "
                   "it is: %s"%repMap["workflow"])
            raise AllInOneError(msg)
        return repMap
