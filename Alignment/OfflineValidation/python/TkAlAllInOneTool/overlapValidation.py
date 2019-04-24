from __future__ import absolute_import

import os

from . import configTemplates
from .genericValidation import GenericValidationData_CTSR, ParallelValidation, ValidationWithPlots
from .helperFunctions import replaceByMap
from .presentation import SubsectionFromList, SubsectionOnePage
from .TkAlExceptions import AllInOneError


class OverlapValidation(GenericValidationData_CTSR, ParallelValidation, ValidationWithPlots):
    configBaseName = "TkAlOverlapValidation"
    scriptBaseName = "TkAlOverlapValidation"
    crabCfgBaseName = "TkAlOverlapValidation"
    resultBaseName = "OverlapValidation"
    outputBaseName = "OverlapValidation"
    mandatories = {"trackcollection"}
    valType = "overlap"

    @property
    def ValidationTemplate(self):
        return configTemplates.overlapTemplate

    @property
    def ValidationSequence(self):
        return configTemplates.overlapValidationSequence

    @property
    def ProcessName(self):
        return "overlap"

    def getRepMap( self, alignment = None ):
        repMap = super(OverlapValidation, self).getRepMap(alignment)
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"],
        })
        return repMap

    def appendToPlots(self):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        return '("{file}", "{title}", {color}, {style}),'.format(file=self.getCompareStrings(plain=True)["DEFAULT"], **self.getRepMap())

    def appendToMerge(self):
        repMap = self.getRepMap()

        parameters = " ".join(os.path.join("root://eoscms//eos/cms", file.lstrip("/")) for file in repMap["resultFiles"])

        mergedoutputfile = os.path.join("root://eoscms//eos/cms", repMap["finalResultFile"].lstrip("/"))
        return "hadd -f %s %s" % (mergedoutputfile, parameters)

    @classmethod
    def plottingscriptname(cls):
        return "TkAlOverlapValidation.py"

    @classmethod
    def plottingscripttemplate(cls):
        return configTemplates.overlapPlottingTemplate

    @classmethod
    def plotsdirname(cls):
        return "OverlapValidationPlots"

    @classmethod
    def runPlots(cls, validations):
        return ("rfcp .oO[plottingscriptpath]Oo. .\n"
                "python .oO[plottingscriptname]Oo.")

