import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap, addIndex
from TkAlExceptions import AllInOneError


class OfflineValidation(GenericValidationData):
    def __init__(self, valName, alignment, config, addDefaults = {}, addMandatories = [],
                 configBaseName = "TkAlOfflineValidation", scriptBaseName = "TkAlOfflineValidation", crabCfgBaseName = "TkAlOfflineValidation",
                 resultBaseName = "AlignmentValidation", outputBaseName = "AlignmentValidation"):
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
        self.configBaseName = configBaseName
        self.scriptBaseName = scriptBaseName
        self.crabCfgBaseName = crabCfgBaseName
        self.resultBaseName = resultBaseName
        self.outputBaseName = outputBaseName
        self.needParentFiles = False
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "offline", addDefaults=defaults,
                                       addMandatories=mandatories)
    
    def createConfiguration(self, path):
        cfgName = "%s.%s.%s_cfg.py"%( self.configBaseName, self.name,
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

    def createScript(self, path):
        return GenericValidationData.createScript(self, path)


    def createCrabCfg(self, path):
        return GenericValidationData.createCrabCfg(self, path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        repMap = GenericValidationData.getRepMap(self, alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "TrackSelectionTemplate": configTemplates.TrackSelectionTemplate,
            "LorentzAngleTemplate": configTemplates.LorentzAngleTemplate,
            "offlineValidationMode": "Standalone",
            "offlineValidationFileOutput": configTemplates.offlineFileOutputTemplate,
            "TrackCollection": self.general["trackcollection"],
            })

        return repMap

    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        if validationsSoFar == "":
            validationsSoFar = ('PlotAlignmentValidation p("root://eoscms//eos/cms%(finalResultFile)s",'
                                '"%(title)s", %(color)s, %(style)s);\n')%repMap
        else:
            validationsSoFar += ('  p.loadFileList("root://eoscms//eos/cms%(finalResultFile)s", "%(title)s",'
                                 '%(color)s, %(style)s);\n')%repMap
        return validationsSoFar

    def appendToMerge( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned,
        else the validation is appended to the list
        """
        repMap = self.getRepMap()

        parameters = "root://eoscms//eos/cms" + ",root://eoscms//eos/cms".join(repMap["resultFiles"])

        mergedoutputfile = "root://eoscms//eos/cms.oO[finalResultFile]Oo."
        validationsSoFar += ('root -x -b -q -l "TkAlOfflineJobsMerge.C(\\\"'
                             +parameters+'\\\",\\\"'+mergedoutputfile+'\\\")"'
                             +"\n")
        return validationsSoFar

class OfflineValidationDQM(OfflineValidation):
    def __init__(self, valName, alignment, config, configBaseName = "TkAlOfflineValidationDQM"):
        OfflineValidation.__init__(self, valName, alignment, config,
                                   configBaseName = configBaseName)
        if not config.has_section("DQM"):
            msg = "You need to have a DQM section in your configfile!"
            raise AllInOneError(msg)
        
        self.__PrimaryDataset = config.get("DQM", "primaryDataset")
        self.__firstRun = int(config.get("DQM", "firstRun"))
        self.__lastRun = int(config.get("DQM", "lastRun"))

    def createConfiguration(self, path):
        OfflineValidation.createConfiguration(self, path)
        
    def createScript(self, path):
        return OfflineValidation.createScript(self, path)

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
