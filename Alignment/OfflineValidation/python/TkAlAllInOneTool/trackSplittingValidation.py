import os
import configTemplates
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class TrackSplittingValidation(GenericValidationData):
    def __init__(self, valName, alignment, config,
                 configBaseName = "TkAlTrackSplitting", scriptBaseName = "TkAlTrackSplitting", crabCfgBaseName = "TkAlTrackSplitting",
                 resultBaseName = "TrackSplitting", outputBaseName = "TrackSplitting"):
        mandatories = ["trackcollection"]
        self.configBaseName = configBaseName
        self.scriptBaseName = scriptBaseName
        self.crabCfgBaseName = crabCfgBaseName
        self.resultBaseName = resultBaseName
        self.outputBaseName = outputBaseName
        self.needParentFiles = False
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "split", addMandatories = mandatories)

    def createConfiguration(self, path ):
        cfgName = "%s.%s.%s_cfg.py"%(self.configBaseName, self.name,
                                     self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.TrackSplittingTemplate}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["finalResultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path):
        return GenericValidationData.createScript(self, path)

    def createCrabCfg(self, path):
        return GenericValidationData.createCrabCfg(self, path, self.crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidationData.getRepMap(self)
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"]
            })
        # repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        # if self.jobmode.split( ',' )[0] == "crab":
        #     repMap["outputFile"] = os.path.basename( repMap["outputFile"] )
        return repMap


    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        comparestring = self.getCompareStrings("TrackSplittingValidation")
        if validationsSoFar != "":
            validationsSoFar += ','
        validationsSoFar += comparestring
        return validationsSoFar

    def appendToMerge( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned,
        else the validation is appended to the list
        """
        repMap = self.getRepMap()

        parameters = " ".join(os.path.join("root://eoscms//eos/cms", file.lstrip("/")) for file in repMap["resultFiles"])

        mergedoutputfile = os.path.join("root://eoscms//eos/cms", repMap["finalResultFile"].lstrip("/"))
        validationsSoFar += "hadd %s %s\n" % (mergedoutputfile, parameters)
        return validationsSoFar
