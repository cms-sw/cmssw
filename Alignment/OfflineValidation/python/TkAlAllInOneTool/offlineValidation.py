import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap, addIndex
from TkAlExceptions import AllInOneError


class OfflineValidation(GenericValidationData):
    def __init__(self, valName, alignment, config, addDefaults = {}, addMandatories = []):
        defaults = {
            "DMRMethod":"median,rmsNorm",
            "DMRMinimum":"30",
            "DMROptions":"",
            "offlineModuleLevelHistsTransient":"False",
            "offlineModuleLevelProfiles":"False",
            "OfflineTreeBaseDir":"TrackHitFilter",
            "SurfaceShapes":"none"
            }
        mandatories = [ "trackcollection" ]
        defaults.update(addDefaults)
        mandatories += addMandatories
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "offline", addDefaults=defaults,
                                       addMandatories=mandatories)
    
    def createConfiguration(self, path,
                            configBaseName = "TkAlOfflineValidation" ):
        cfgName = "%s.%s.%s_cfg.py"%( configBaseName, self.name,
                                      self.alignmentToValidate.name )
        repMap = self.getRepMap()
        if self.NJobs > 1 and self.general["offlineModuleLevelHistsTransient"] == "True":
            msg = ("To be able to merge results when running parallel jobs,"
                   " set offlineModuleLevelHistsTransient to false.")
            raise AllInOneError(msg)

        templateToUse = configTemplates.offlineTemplate
        if self.AutoAlternates:
            if "Cosmics" in self.general["trackcollection"]:
                Bfield = self.dataset.magneticFieldForRun()
                if Bfield > 3.3 and Bfield < 4.3:                 #Should never be 4.3, but this covers strings, which always compare bigger than ints
                    templateToUse = configTemplates.CosmicsOfflineValidation
                    print ("B field for %s = %sT.  Using the template for cosmics at 3.8T.\n"
                           "To override this behavior, specify AutoAlternates = false in the [alternateTemplates] section") % (self.dataset.name(), Bfield)
                elif Bfield < 0.5:
                    templateToUse = configTemplates.CosmicsAt0TOfflineValidation
                    print ("B field for %s = %sT.  Using the template for cosmics at 0T.\n"
                           "To override this behavior, specify AutoAlternates = false in the [alternateTemplates] section") % (self.dataset.name(), Bfield)
                else:
                    try:
                        if "unknown " in Bfield:
                            msg = Bfield.replace("unknown ","",1)
                        elif "Bfield" is "unknown":
                            msg = "Can't get the B field for %s." % self.dataset.name()
                    except TypeError:
                        msg = "B field for %s = %sT.  This is not that close to 0T or 3.8T." % (self.dataset.name(), Bfield)
                    raise AllInOneError(msg + "\n"
                                        "To use this data, turn off the automatic alternates using AutoAlternates = false\n"
                                        "in the [alternateTemplates] section, and choose the alternate template yourself.")

        cfgs = {cfgName: templateToUse}
        self.filesToCompare[
            GenericValidationData.defaultReferenceName ] = repMap["finalResultFile"]
        return GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path, scriptBaseName = "TkAlOfflineValidation"):
        scriptName = "%s.%s.%s.sh"%( scriptBaseName, self.name,
                                     self.alignmentToValidate.name )
        repMap = self.getRepMap()
        repMap["CommandLine"]=""
        for cfg in self.configFiles:
            repMap["CommandLine"] += repMap["CommandLineTemplate"]%{"cfgFile":addIndex(cfg, self.NJobs, ".oO[nIndex]Oo."),
                                                                    "postProcess":""
                                                                   }
        scripts = {scriptName: configTemplates.scriptTemplate}
        return GenericValidationData.createScript(self, scripts, path, repMap = repMap)

    def createCrabCfg(self, path, crabCfgBaseName = "TkAlOfflineValidation"):
        return GenericValidationData.createCrabCfg(self, path, crabCfgBaseName)

    def getRepMap(self, alignment = None):
        repMap = GenericValidationData.getRepMap(self, alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "resultFile": "resultFiles[.oO[nIndex]Oo.]",
            "resultFiles": addIndex(os.path.expandvars(replaceByMap(
                                         "/store/caf/user/$USER/.oO[eosdir]Oo./AlignmentValidation_" + self.name + "_.oO[name]Oo..root"
                                                   , repMap)), self.NJobs),
            "finalResultFile": os.path.expandvars(replaceByMap(
                                    "/store/caf/user/$USER/.oO[eosdir]Oo./AlignmentValidation_" + self.name + "_.oO[name]Oo..root"
                                                   , repMap)),
            "TrackSelectionTemplate": configTemplates.TrackSelectionTemplate,
            "LorentzAngleTemplate": configTemplates.LorentzAngleTemplate,
            "offlineValidationMode": "Standalone",
            "offlineValidationFileOutput": configTemplates.offlineFileOutputTemplate,
            "TrackCollection": self.general["trackcollection"],
            "outputFile": ".oO[outputFiles[.oO[nIndex]Oo.]]Oo.",
            "outputFiles": addIndex(os.path.expandvars(replaceByMap(
                                "AlignmentValidation_" + self.name + "_.oO[name]Oo..root"
                                                   , repMap)), self.NJobs),
            "finalOutputFile": os.path.expandvars(replaceByMap(
                                    "AlignmentValidation_" + self.name + "_.oO[name]Oo..root"
                                              , repMap))
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
            validationsSoFar = ('PlotAlignmentValidation p("%(finalOutputFile)s",'
                                '"%(name)s", %(color)s, %(style)s);\n')%repMap
        else:
            validationsSoFar += ('p.loadFileList("%(finalOutputFile)s", "%(name)s",'
                                 '%(color)s, %(style)s);\n')%repMap
        return validationsSoFar

    def appendToMerge( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned,
        else the validation is appended to the list
        """
        repMap = self.getRepMap()

        parameters = ",".join(repMap["outputFiles"])

        mergedoutputfile = repMap["finalOutputFile"]
        validationsSoFar += ('root -x -b -q -l "TkAlOfflineJobsMerge.C(\\\"'
                             +parameters+'\\\",\\\"'+mergedoutputfile+'\\\")"'
                             +"\n")
        return validationsSoFar

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
