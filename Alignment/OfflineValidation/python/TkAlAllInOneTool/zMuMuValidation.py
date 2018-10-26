import os
import configTemplates
import globalDictionaries
from genericValidation import GenericValidationData, ValidationWithPlots
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class ZMuMuValidation(GenericValidationData, ValidationWithPlots):
    configBaseName = "TkAlZMuMuValidation"
    scriptBaseName = "TkAlZMuMuValidation"
    crabCfgBaseName = "TkAlZMuMuValidation"
    resultBaseName = "ZMuMuValidation"
    outputBaseName = "ZMuMuValidation"
    defaults = {
        "zmumureference": ("/store/caf/user/emiglior/Alignment/TkAlDiMuonValidation/Reference/BiasCheck_DYToMuMu_Summer12_TkAlZMuMu_IDEAL.root"),
        }
    deprecateddefaults = {
        "resonance": "",
        "switchONfit": "",
        "rebinphi": "",
        "rebinetadiff": "",
        "rebineta": "",
        "rebinpt": "",
        }
    defaults.update(deprecateddefaults)
    needpackages = {'MuonAnalysis/MomentumScaleCalibration'}
    mandatories = {"etamaxneg", "etaminneg", "etamaxpos", "etaminpos"}
    valType = "zmumu"
    def __init__(self, valName, alignment, config):
        super(ZMuMuValidation, self).__init__(valName, alignment, config)
        if self.general["zmumureference"].startswith("/store"):
            self.general["zmumureference"] = "root://eoscms//eos/cms" + self.general["zmumureference"]
        if self.NJobs > 1:
            raise AllInOneError("Parallel jobs not implemented for the Z->mumu validation!\n"
                                "Please set parallelJobs = 1.")
        for option in self.deprecateddefaults:
            if self.general[option]:
                raise AllInOneError("The '%s' option has been moved to the [plots:zmumu] section.  Please specify it there."%option)
            del self.general[option]

    @property
    def filesToCompare(self):
        return {self.defaultReferenceName: replaceByMap(".oO[eosdir]Oo./0_zmumuHisto.root", self.getRepMap())}

    @property
    def ValidationTemplate(self):
        return configTemplates.ZMuMuValidationTemplate

    @property
    def ProcessName(self):
        return "ONLYHISTOS"

    @property
    def FileOutputTemplate(self):
        return ""

    @property
    def LoadBasicModules(self):
        return super(ZMuMuValidation, self).LoadBasicModules + configTemplates.LoadMuonModules

    @property
    def TrackSelectionRefitting(self):
        return configTemplates.SingleTrackRefitter

    @property
    def DefinePath(self):
        return configTemplates.ZMuMuPath

    def createScript(self, path):
        return super(ZMuMuValidation, self).createScript(path, template = configTemplates.zMuMuScriptTemplate)

    def createCrabCfg(self, path):
        return super(ZMuMuValidation, self).createCrabCfg(path, self.crabCfgBaseName)

    def getRepMap(self, alignment = None):
        if alignment == None:
            alignment = self.alignmentToValidate
        repMap = super(ZMuMuValidation, self).getRepMap(alignment)
        repMap.update({
            "nEvents": self.general["maxevents"],
            "outputFile": ("0_zmumuHisto.root"
                           ",genSimRecoPlots.root"
                           ",FitParameters.txt"),
            "eosdir": os.path.join(self.general["eosdir"], "%s/%s/%s" % (self.outputBaseName, self.name, alignment.name)),
            "workingdir": ".oO[datadir]Oo./%s/%s/%s" % (self.outputBaseName, self.name, alignment.name),
            "plotsdir": ".oO[datadir]Oo./%s/%s/%s/plots" % (self.outputBaseName, self.name, alignment.name),
            "TrackCollection": self.trackcollection,
        })
        return repMap

    @property
    def trackcollection(self):
        from plottingOptions import PlottingOptions
        resonance = PlottingOptions(self.config, self.valType)["resonance"]
        if resonance == "Z":
            return 'ALCARECOTkAlZMuMu'
        elif resonance == "JPsi":
            return 'ALCARECOTkAlJpsiMuMu'
        elif resonance in ("Y1S", "Y2S", "Y3S"):
            return 'ALCARECOTkAlUpsilonMuMu'
        else:
            raise AllInOneError("Unknown resonance {}!".format(resonance))

    def appendToPlots(self):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        return replaceByMap('    filenames.push_back("root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./BiasCheck.root");  titles.push_back(".oO[title]Oo.");  colors.push_back(.oO[color]Oo.);  linestyles.push_back(.oO[style]Oo.);\n', repMap)

    @classmethod
    def plottingscriptname(cls):
        return "TkAlMergeZmumuPlots.C"

    @classmethod
    def plottingscripttemplate(cls):
        return configTemplates.mergeZmumuPlotsTemplate

    @classmethod
    def plotsdirname(cls):
        return ".oO[resonance]Oo.MuMuPlots"
