import os
import configTemplates
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class TrackSplittingValidation(GenericValidationData):
    configBaseName = "TkAlTrackSplitting"
    scriptBaseName = "TkAlTrackSplitting"
    crabCfgBaseName = "TkAlTrackSplitting"
    resultBaseName = "TrackSplitting"
    outputBaseName = "TrackSplitting"
    mandatories = {"trackcollection"}
    def __init__(self, valName, alignment, config):
        super(TrackSplittingValidation, self).__init__(valName, alignment, config,
                                                       "split")

    @property
    def cfgTemplate(self):
        return configTemplates.TrackSplittingTemplate

    def createScript(self, path):
        return super(TrackSplittingValidation, self).createScript(path)

    def createCrabCfg(self, path):
        return super(TrackSplittingValidation, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = super(TrackSplittingValidation, self).getRepMap()
        if repMap["subdetector"] == "none":
            subdetselection = ""
        else:
            subdetselection = "process.AlignmentTrackSelector.minHitsPerSubDet.in.oO[subdetector]Oo. = 2"
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"],
            "subdetselection": subdetselection,
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
            validationsSoFar += ',"\n              "'
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
        validationsSoFar += "hadd -f %s %s\n" % (mergedoutputfile, parameters)
        return validationsSoFar
