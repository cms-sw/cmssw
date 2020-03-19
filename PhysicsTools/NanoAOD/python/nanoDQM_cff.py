import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *

## Modify plots accordingly to era
_vplots80X = nanoDQM.vplots.clone()
# Tau plots
_tauPlots80X = cms.VPSet()
for plot in _vplots80X.Tau.plots:
    if (plot.name.value().find("MVA")>-1 and plot.name.value().find("2017")>-1) or (plot.name.value().find("AntiEle")>-1 and plot.name.value().find("2018")>-1):
        continue
    _tauPlots80X.append(plot)
_tauPlots80X.append(Plot1D('idMVAnewDM', 'idMVAnewDM', 64, -0.5, 63.5, 'IsolationMVArun2v1DBnewDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'))
_tauPlots80X.append(Plot1D('idMVAoldDMdR03', 'idMVAoldDMdR03', 64, -0.5, 63.5, 'IsolationMVArun2v1DBdR03oldDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'))
_tauPlots80X.append(Plot1D('rawMVAnewDM', 'rawMVAnewDM', 20, -1, 1, 'byIsolationMVArun2v1DBnewDMwLT raw output discriminator'))
_tauPlots80X.append(Plot1D('rawMVAoldDMdR03', 'rawMVAoldDMdR03', 20, -1, 1, 'byIsolationMVArun2v1DBdR03oldDMwLT raw output discriminator'))
_vplots80X.Tau.plots = _tauPlots80X
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(nanoDQM,
                                     vplots = _vplots80X
)

_METFixEE2017_DQMentry = nanoDQM.vplots.MET.clone()
_METFixEE2017_plots = cms.VPSet()
for plot in _METFixEE2017_DQMentry.plots:
    if plot.name.value().find("fiducial")>-1: continue
    _METFixEE2017_plots.append(plot)
_METFixEE2017_DQMentry.plots = _METFixEE2017_plots
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv1_cff import run2_nanoAOD_94XMiniAODv1
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv2_cff import run2_nanoAOD_94XMiniAODv2
for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
    modifier.toModify(nanoDQM.vplots, METFixEE2017 = _METFixEE2017_DQMentry)

_Electron_plots_2016 = copy.deepcopy(nanoDQM.vplots.Electron.plots)
_Electron_plots_2016.append(Plot1D('cutBased_HLTPreSel', 'cutBased_HLTPreSel', 2, -0.5, 1.5, 'cut-based HLT pre-selection ID'))
_Electron_plots_2016.append(Plot1D('cutBased_Spring15', 'cutBased_Spring15', 5, -0.5, 4.5, 'cut-based Spring15 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)'))
_Electron_plots_2016.append(Plot1D('mvaSpring16GP', 'mvaSpring16GP', 20, -1, 1, 'MVA Spring16 general-purpose ID score'))
_Electron_plots_2016.append(Plot1D('mvaSpring16GP_WP80', 'mvaSpring16GP_WP80', 2, -0.5, 1.5, 'MVA Spring16 general-purpose ID WP80'))
_Electron_plots_2016.append(Plot1D('mvaSpring16GP_WP90', 'mvaSpring16GP_WP90', 2, -0.5, 1.5, 'MVA Spring16 general-purpose ID WP90'))
_Electron_plots_2016.append(Plot1D('mvaSpring16HZZ', 'mvaSpring16HZZ', 20, -1, 1, 'MVA Spring16 HZZ ID score'))
_Electron_plots_2016.append(Plot1D('mvaSpring16HZZ_WPL', 'mvaSpring16HZZ_WPL', 2, -0.5, 1.5, 'MVA Spring16 HZZ ID loose WP'))
_Electron_plots_2016.append(NoPlot('vidNestedWPBitmapSpring15'))

_Photon_plots_2016 = copy.deepcopy(nanoDQM.vplots.Photon.plots)
_Photon_plots_2016.append(Plot1D('cutBased', 'cutBased', 4, -0.5, 3.5, 'cut-based Spring16-V2p2 ID (0:fail, 1::loose, 2:medium, 3:tight)'))
_Photon_plots_2016.append(Plot1D('cutBased17Bitmap', 'cutBased17Bitmap', 8, -0.5, 7.5, 'cut-based Fall17-94X-V1 ID bitmap, 2^(0:loose, 1:medium, 2:tight)'))
_Photon_plots_2016.append(Plot1D('mvaID17', 'mvaID17', 20, -1, 1, 'MVA Fall17v1p1 ID score'))
_Photon_plots_2016.append(Plot1D('mvaID17_WP80', 'mvaID17_WP80', 2, -0.5, 1.5, 'MVA Fall17v1p1 ID WP80'))
_Photon_plots_2016.append(Plot1D('mvaID17_WP90', 'mvaID17_WP90', 2, -0.5, 1.5, 'MVA Fall17v1p1 ID WP90'))

_FatJet_plots_80x = copy.deepcopy(nanoDQM.vplots.FatJet.plots)
_FatJet_plots_80x.append(Plot1D('msoftdrop_chs', 'msoftdrop_chs', 20, -300, 300, 'Legacy uncorrected soft drop mass with CHS'))

_Flag_plots_80x = copy.deepcopy(nanoDQM.vplots.Flag.plots)
_Flag_plots_80x.append(Plot1D('BadGlobalMuon', 'BadGlobalMuon', 2, -0.5, 1.5, 'Bad muon flag'))
_Flag_plots_80x.append(Plot1D('CloneGlobalMuon', 'CloneGlobalMuon', 2, -0.5, 1.5, 'Clone muon flag'))

from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016
for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
    modifier.toModify(nanoDQM.vplots.Electron, plots = _Electron_plots_2016)
    modifier.toModify(nanoDQM.vplots.Photon, plots = _Photon_plots_2016)
run2_miniAOD_80XLegacy.toModify(nanoDQM.vplots.FatJet, plots = _FatJet_plots_80x)
run2_miniAOD_80XLegacy.toModify(nanoDQM.vplots.Flag, plots = _Flag_plots_80x)

run2_miniAOD_80XLegacy.toModify(nanoDQM.vplots, IsoTrack = None)

## MC
nanoDQMMC = nanoDQM.clone()
nanoDQMMC.vplots.Electron.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.Muon.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.Photon.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.Tau.sels.Prompt = cms.string("genPartFlav == 5")
nanoDQMMC.vplots.Jet.sels.Prompt = cms.string("genJetIdx != 1")
nanoDQMMC.vplots.Jet.sels.PromptB = cms.string("genJetIdx != 1 && hadronFlavour == 5")

from DQMServices.Core.DQMQualityTester import DQMQualityTester
nanoDQMQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('PhysicsTools/NanoAOD/test/dqmQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True)
)

nanoHarvest = cms.Sequence( nanoDQMQTester )
