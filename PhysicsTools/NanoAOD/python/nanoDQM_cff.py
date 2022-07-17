import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

## Modify plots accordingly to era
_vplots80X = nanoDQM.vplots.clone()
# Tau plots
_tauPlots80X = cms.VPSet()
for plot in _vplots80X.Tau.plots:
    if (plot.name.value().find("MVA")>-1 and plot.name.value().find("2017")>-1) or (plot.name.value().find("AntiEle")>-1 and plot.name.value().find("2018")>-1) or (plot.name.value().find("AntiEleDeadECal")>-1) or (plot.name.value().find("DeepTau")>-1):
        continue
    _tauPlots80X.append(plot)
_tauPlots80X.extend([Plot1D('idMVAnewDM', 'idMVAnewDM', 64, -0.5, 63.5, 'IsolationMVArun2v1DBnewDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'),
                     Plot1D('idMVAoldDM', 'idMVAoldDM', 64, -0.5, 63.5, 'IsolationMVArun2v1DBnewDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'),
                     Plot1D('idMVAoldDMdR03', 'idMVAoldDMdR03', 64, -0.5, 63.5, 'IsolationMVArun2v1DBdR03oldDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'),
                     Plot1D('rawMVAnewDM', 'rawMVAnewDM', 20, -1, 1, 'byIsolationMVArun2v1DBnewDMwLT raw output discriminator'),
                     Plot1D('rawMVAoldDM', 'rawMVAoldDM', 20, -1, 1, 'byIsolationMVArun2v1DBnewDMwLT raw output discriminator'),
                     Plot1D('rawMVAoldDMdR03', 'rawMVAoldDMdR03', 20, -1, 1, 'byIsolationMVArun2v1DBdR03oldDMwLT raw output discriminator'),
                     Plot1D('idAntiEle', 'idAntiEle', 32, -0.5, 31.5, 'Anti-electron MVA discriminator V6: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight'),
                     Plot1D('rawAnti', 'rawAntiEle', 20, -100, 100, 'Anti-electron MVA discriminator V6 raw output discriminator'),
                     Plot1D('rawAntiEleCat', 'rawAntiEleCat', 17, -1.5, 15.5, 'Anti-electron MVA discriminator V6 category')
])
_vplots80X.Tau.plots = _tauPlots80X
run2_miniAOD_80XLegacy.toModify(nanoDQM,
                                vplots = _vplots80X
)
_tauPlotsPreV9 = cms.VPSet()
for plot in nanoDQM.vplots.Tau.plots:
    if plot.name.value()!="idDecayModeOldDMs":
        _tauPlotsPreV9.append(plot)
_tauPlotsPreV9.extend([
                Plot1D('idDecayMode', 'idDecayMode', 2, -0.5, 1.5, "tauID('decayModeFinding')"),
                Plot1D('idDecayModeNewDMs', 'idDecayModeNewDMs', 2, -0.5, 1.5, "tauID('decayModeFindingNewDMs')"),
                Plot1D('idAntiEle', 'idAntiEle', 32, -0.5, 31.5, 'Anti-electron MVA discriminator V6: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight'),
                Plot1D('idAntiEle2018', 'idAntiEle2018', 32, -0.5, 31.5, 'Anti-electron MVA discriminator V6 (2018): bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight'),
                Plot1D('idMVAnewDM2017v2', 'idMVAnewDM2017v2', 128, -0.5, 127.5, 'IsolationMVArun2v1DBnewDMwLT ID working point (2017v2): bitmask 1 = VVLoose, 2 = VLoose, 4 = Loose, 8 = Medium, 16 = Tight, 32 = VTight, 64 = VVTight'),
                Plot1D('idMVAoldDM', 'idMVAoldDM', 64, -0.5, 63.5, 'IsolationMVArun2v1DBoldDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'),
                Plot1D('idMVAoldDM2017v1', 'idMVAoldDM2017v1', 128, -0.5, 127.5, 'IsolationMVArun2v1DBoldDMwLT ID working point (2017v1): bitmask 1 = VVLoose, 2 = VLoose, 4 = Loose, 8 = Medium, 16 = Tight, 32 = VTight, 64 = VVTight'),
                Plot1D('idMVAoldDM2017v2', 'idMVAoldDM2017v2', 128, -0.5, 127.5, 'IsolationMVArun2v1DBoldDMwLT ID working point (2017v2): bitmask 1 = VVLoose, 2 = VLoose, 4 = Loose, 8 = Medium, 16 = Tight, 32 = VTight, 64 = VVTight'),
                Plot1D('idMVAoldDMdR032017v2', 'idMVAoldDMdR032017v2', 128, -0.5, 127.5, 'IsolationMVArun2v1DBdR03oldDMwLT ID working point (217v2): bitmask 1 = VVLoose, 2 = VLoose, 4 = Loose, 8 = Medium, 16 = Tight, 32 = VTight, 64 = VVTight'),
                Plot1D('rawAntiEle', 'rawAntiEle', 20, -100, 100, 'Anti-electron MVA discriminator V6 raw output discriminator'),
                Plot1D('rawAntiEle2018', 'rawAntiEle2018', 20, -100, 100, 'Anti-electron MVA discriminator V6 raw output discriminator (2018)'),
                Plot1D('rawAntiEleCat', 'rawAntiEleCat', 17, -1.5, 15.5, 'Anti-electron MVA discriminator V6 category'),
                Plot1D('rawAntiEleCat2018', 'rawAntiEleCat2018', 17, -1.5, 15.5, 'Anti-electron MVA discriminator V6 category (2018)'),
                Plot1D('rawMVAnewDM2017v2', 'rawMVAnewDM2017v2', 20, -1, 1, 'byIsolationMVArun2v1DBnewDMwLT raw output discriminator (2017v2)'),
                Plot1D('rawMVAoldDM', 'rawMVAoldDM', 20, -1, 1, 'byIsolationMVArun2v1DBoldDMwLT raw output discriminator'),
                Plot1D('rawMVAoldDM2017v1', 'rawMVAoldDM2017v1', 20, -1, 1, 'byIsolationMVArun2v1DBoldDMwLT raw output discriminator (2017v1)'),
                Plot1D('rawMVAoldDM2017v2', 'rawMVAoldDM2017v2', 20, -1, 1, 'byIsolationMVArun2v1DBoldDMwLT raw output discriminator (2017v2)'),
                Plot1D('rawMVAoldDMdR032017v2', 'rawMVAoldDMdR032017v2', 20, -1, 1, 'byIsolationMVArun2v1DBdR03oldDMwLT raw output discriminator (2017v2)')
])

(run2_nanoAOD_92X | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_94X2016 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1).toModify(nanoDQM.vplots.Tau, plots = _tauPlotsPreV9)

_boostedTauPlotsV10 = cms.VPSet()
for plot in nanoDQM.vplots.boostedTau.plots:
    _boostedTauPlotsV10.append(plot)
_boostedTauPlotsV10.extend([
    Plot1D('idMVAoldDMdR032017v2', 'idMVAoldDMdR032017v2', 128, -0.5, 127.5, 'IsolationMVArun2017v2DBoldDMdR0p3wLT ID working point (2017v2): bitmask 1 = VVLoose, 2 = VLoose, 4 = Loose, 8 = Medium, 16 = Tight, 32 = VTight, 64 = VVTight'),
    Plot1D('rawMVAoldDMdR032017v2', 'rawMVAoldDMdR032017v2', 20, -1, 1, 'byIsolationMVArun2017v2DBoldDMdR0p3wLT raw output discriminator (2017v2)')
])

(run2_nanoAOD_106Xv2).toModify(nanoDQM.vplots.boostedTau, plots = _boostedTauPlotsV10)

_METFixEE2017_DQMentry = nanoDQM.vplots.MET.clone()
_METFixEE2017_plots = cms.VPSet()
for plot in _METFixEE2017_DQMentry.plots:
    if plot.name.value().find("fiducial")>-1: continue
    _METFixEE2017_plots.append(plot)
_METFixEE2017_DQMentry.plots = _METFixEE2017_plots
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

#putting back the fall17V1 plots for non v9 case
_Electron_plots_withFall17V1 = copy.deepcopy(nanoDQM.vplots.Electron.plots)
_Electron_plots_withFall17V1.append(Plot1D('cutBased_Fall17_V1', 'cutBased_Fall17_V1', 5, -0.5, 4.5, 'cut-based ID Fall17 V1 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1Iso', 'mvaFall17V1Iso', 20, -1, 1, 'MVA Iso ID V1 score'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1Iso_WP80', 'mvaFall17V1Iso_WP80', 2, -0.5, 1.5, 'MVA Iso ID V1 WP80'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1Iso_WP90', 'mvaFall17V1Iso_WP90', 2, -0.5, 1.5, 'MVA Iso ID V1 WP90'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1Iso_WPL', 'mvaFall17V1Iso_WPL', 2, -0.5, 1.5, 'MVA Iso ID V1 loose WP'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1noIso', 'mvaFall17V1noIso', 20, -1, 1, 'MVA noIso ID V1 score'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1noIso_WP80', 'mvaFall17V1noIso_WP80', 2, -0.5, 1.5, 'MVA noIso ID V1 WP80'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1noIso_WP90', 'mvaFall17V1noIso_WP90', 2, -0.5, 1.5, 'MVA noIso ID V1 WP90'))
_Electron_plots_withFall17V1.append(Plot1D('mvaFall17V1noIso_WPL', 'mvaFall17V1noIso_WPL', 2, -0.5, 1.5, 'MVA noIso ID V1 loose WP'))

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

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
    modifier.toModify(nanoDQM.vplots.Electron, plots = _Electron_plots_2016)
    modifier.toModify(nanoDQM.vplots.Photon, plots = _Photon_plots_2016)
run2_miniAOD_80XLegacy.toModify(nanoDQM.vplots.FatJet, plots = _FatJet_plots_80x)
run2_miniAOD_80XLegacy.toModify(nanoDQM.vplots.Flag, plots = _Flag_plots_80x)
(run2_nanoAOD_92X | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_94X2016 | run2_nanoAOD_102Xv1).toModify(nanoDQM.vplots.Electron, plots=_Electron_plots_withFall17V1)

run2_miniAOD_80XLegacy.toModify(nanoDQM.vplots, IsoTrack = None)

## MC
nanoDQMMC = nanoDQM.clone()
nanoDQMMC.vplots.Electron.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.LowPtElectron.sels.Prompt = cms.string("genPartFlav == 1")
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

(run3_nanoAOD_devel).toModify(nanoDQM.vplots, Electron = None)
(run3_nanoAOD_devel).toModify(nanoDQMMC.vplots, Electron = None)

_modifiers = ( run2_miniAOD_80XLegacy |
               run2_nanoAOD_94XMiniAODv1 |
               run2_nanoAOD_94XMiniAODv2 |
               run2_nanoAOD_94X2016 |
               run2_nanoAOD_102Xv1 |
               run2_nanoAOD_106Xv1 )
_modifiers.toModify(nanoDQM.vplots, LowPtElectron = None)
_modifiers.toModify(nanoDQMMC.vplots, LowPtElectron = None)

nanoHarvest = cms.Sequence( nanoDQMQTester )
