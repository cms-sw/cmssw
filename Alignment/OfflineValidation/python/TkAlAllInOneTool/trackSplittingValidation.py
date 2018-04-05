import os
import configTemplates
from genericValidation import GenericValidationData_CTSR, ParallelValidation, ValidationForPresentation, ValidationWithPlotsSummary
from helperFunctions import replaceByMap
from presentation import SubsectionFromList, SubsectionOnePage
from TkAlExceptions import AllInOneError


class TrackSplittingValidation(GenericValidationData_CTSR, ParallelValidation, ValidationWithPlotsSummary, ValidationForPresentation):
    configBaseName = "TkAlTrackSplitting"
    scriptBaseName = "TkAlTrackSplitting"
    crabCfgBaseName = "TkAlTrackSplitting"
    resultBaseName = "TrackSplitting"
    outputBaseName = "TrackSplitting"
    mandatories = {"trackcollection"}
    valType = "split"

    @property
    def ValidationTemplate(self):
        return configTemplates.TrackSplittingTemplate

    @property
    def ValidationSequence(self):
        return configTemplates.TrackSplittingSequence

    @property
    def ProcessName(self):
        return "splitter"

    def createScript(self, path):
        return super(TrackSplittingValidation, self).createScript(path)

    def createCrabCfg(self, path):
        return super(TrackSplittingValidation, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = super(TrackSplittingValidation, self).getRepMap(alignment)
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

    def appendToPlots(self):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        comparestring = self.getCompareStrings("TrackSplittingValidation")
        return '              "{},"'.format(comparestring)

    def appendToMerge(self):
        repMap = self.getRepMap()

        parameters = " ".join(os.path.join("root://eoscms//eos/cms", file.lstrip("/")) for file in repMap["resultFiles"])

        mergedoutputfile = os.path.join("root://eoscms//eos/cms", repMap["finalResultFile"].lstrip("/"))
        return "hadd -f %s %s" % (mergedoutputfile, parameters)

    @classmethod
    def plottingscriptname(cls):
        return "TkAlTrackSplitPlot.C"

    @classmethod
    def plottingscripttemplate(cls):
        return configTemplates.trackSplitPlotTemplate

    @classmethod
    def plotsdirname(cls):
        return "TrackSplittingPlots"

    @classmethod
    def presentationsubsections(cls):
        return [
            SubsectionTrackSplitting('hist.*eps$', 'Track splitting'),
#           Uncomment and edit to highlight one or more profiles
#            SubsectionOnePage("profile.phi_org.Delta_phi.*.eps", "modulation"),
        ]

class SubsectionTrackSplitting(SubsectionFromList):
    pageidentifiers = (
                       ("hist[.]Delta_pt", "$p_T$"),
                       ("hist[.]Delta_(eta|phi)", "Angles"),
                       ("hist[.]Delta_d(xy|z)", "Vertex"),
                      )

