from __future__ import absolute_import
import os
from . import configTemplates
from . import globalDictionaries
from .genericValidation import GenericValidationData_CTSR, ParallelValidation, ValidationWithComparison, ValidationForPresentation, ValidationWithPlots, ValidationWithPlotsSummary
from .helperFunctions import replaceByMap, addIndex, pythonboolstring
from .presentation import SubsectionFromList, SubsectionOnePage
from .TkAlExceptions import AllInOneError

class OfflineValidation(GenericValidationData_CTSR, ParallelValidation, ValidationWithComparison, ValidationWithPlotsSummary, ValidationForPresentation):
    configBaseName = "TkAlOfflineValidation"
    scriptBaseName = "TkAlOfflineValidation"
    crabCfgBaseName = "TkAlOfflineValidation"
    resultBaseName = "AlignmentValidation"
    outputBaseName = "AlignmentValidation"
    defaults = {
        "offlineModuleLevelHistsTransient": "False",
        "offlineModuleLevelProfiles": "True",
        "stripYResiduals": "False",
        "maxtracks": "0",
        "chargeCut": "0",
        }
    deprecateddefaults = {
        "DMRMethod":"",
        "DMRMinimum":"",
        "DMROptions":"",
        "OfflineTreeBaseDir":"",
        "SurfaceShapes":"",
        }
    defaults.update(deprecateddefaults)
    mandatories = {"trackcollection"}
    valType = "offline"

    def __init__(self, valName, alignment, config):
        super(OfflineValidation, self).__init__(valName, alignment, config)

        for name in "offlineModuleLevelHistsTransient", "offlineModuleLevelProfiles", "stripYResiduals":
            self.general[name] = pythonboolstring(self.general[name], name)

        for option in self.deprecateddefaults:
            if self.general[option]:
                raise AllInOneError("The '%s' option has been moved to the [plots:offline] section.  Please specify it there."%option)
            del self.general[option]

        if self.NJobs > 1 and self.general["offlineModuleLevelHistsTransient"] == "True":
            msg = ("To be able to merge results when running parallel jobs,"
                   " set offlineModuleLevelHistsTransient to false.")
            raise AllInOneError(msg)

        try:
            self.NTracks = int(self.general["maxtracks"])
            if self.NTracks < 0: raise ValueError
        except ValueError:
            raise AllInOneError("maxtracks has to be a positive integer, or 0 for no limit")

        if self.NTracks / self.NJobs != float(self.NTracks) / self.NJobs:
            raise AllInOneError("maxtracks has to be divisible by parallelJobs")

    @property
    def ProcessName(self):
        return "OfflineValidator"

    @property
    def ValidationTemplate(self):
        return configTemplates.offlineTemplate

    @property
    def ValidationSequence(self):
        return configTemplates.OfflineValidationSequence

    @property
    def FileOutputTemplate(self):
        return configTemplates.offlineFileOutputTemplate

    def createScript(self, path):
        return super(OfflineValidation, self).createScript(path)

    def createCrabCfg(self, path):
        return super(OfflineValidation, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        repMap = super(OfflineValidation, self).getRepMap(alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "offlineValidationMode": "Standalone",
            "TrackCollection": self.general["trackcollection"],
            "filetoplot": "root://eoscms//eos/cms.oO[finalResultFile]Oo.",
            })

        return repMap

    def appendToPlots(self):
        return '  p.loadFileList(".oO[filetoplot]Oo.", ".oO[title]Oo.", .oO[color]Oo., .oO[style]Oo.);\n'

    @classmethod
    def initMerge(cls):
        from .plottingOptions import PlottingOptions
        outFilePath = replaceByMap(".oO[scriptsdir]Oo./TkAlOfflineJobsMerge.C", PlottingOptions(None, cls.valType))
        with open(outFilePath, "w") as theFile:
            theFile.write(replaceByMap(configTemplates.mergeOfflineParJobsTemplate, {}))
        result = super(OfflineValidation, cls).initMerge()
        result += ("cp .oO[Alignment/OfflineValidation]Oo./scripts/merge_TrackerOfflineValidation.C .\n"
                   "rfcp .oO[mergeOfflineParJobsScriptPath]Oo. .\n")
        return result

    def appendToMerge(self):
        repMap = self.getRepMap()

        parameters = "root://eoscms//eos/cms" + ",root://eoscms//eos/cms".join(repMap["resultFiles"])

        mergedoutputfile = "root://eoscms//eos/cms%(finalResultFile)s"%repMap
        return ('root -x -b -q -l "TkAlOfflineJobsMerge.C(\\\"'
                +parameters+'\\\",\\\"'+mergedoutputfile+'\\\")"')

    @classmethod
    def plottingscriptname(cls):
        return "TkAlExtendedOfflineValidation.C"

    @classmethod
    def plottingscripttemplate(cls):
        return configTemplates.extendedValidationTemplate

    @classmethod
    def plotsdirname(cls):
        return "ExtendedOfflineValidation_Images"

    @classmethod
    def comparealignmentsname(cls):
        return "compareAlignments.cc"

    @classmethod
    def presentationsubsections(cls):
        return [
            SubsectionOnePage('chi2', r'$\chi^2$ plots'),
            SubsectionSubdetectors('DmedianY*R_[^_]*.eps$', 'DMR'),
            SubsectionSubdetectors('DmedianY*R.*plain.eps$', 'DMR'),
            SubsectionSubdetectors('DmedianY*R.*split.eps$','Split DMR'),
            SubsectionSubdetectors('DrmsNY*R_[^_]*.eps$', 'DRnR'),
            SubsectionSubdetectors('DrmsNY*R.*plain.eps$', 'DRnR'),
            SubsectionSubdetectors('SurfaceShape', 'Surface Shape'),
        ]

class SubsectionSubdetectors(SubsectionFromList):
    pageidentifiers = (
                       ("BPIX", "BPIX"),
                       ("FPIX", "FPIX"),
                       ("TIB", "TIB"),
                       ("TID", "TID"),
                       ("TOB", "TOB"),
                       ("TEC", "TEC"),
                      )

class OfflineValidationDQM(OfflineValidation):
    configBaseName = "TkAlOfflineValidationDQM"
    def __init__(self, valName, alignment, config):
        super(OfflineValidationDQM, self).__init__(valName, alignment, config)
        if not config.has_section("DQM"):
            msg = "You need to have a DQM section in your configfile!"
            raise AllInOneError(msg)
        
        self.__PrimaryDataset = config.get("DQM", "primaryDataset")
        self.__firstRun = int(config.get("DQM", "firstRun"))
        self.__lastRun = int(config.get("DQM", "lastRun"))

    def getRepMap(self, alignment = None):
        repMap = super(OfflineValidationDQM, self).getRepMap(alignment)
        repMap.update({
                "workdir": os.path.expandvars(repMap["workdir"]),
		"offlineValidationMode": "Dqm",
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

    @property
    def FileOutputTemplate(self):
        return configTemplates.offlineDqmFileOutputTemplate
