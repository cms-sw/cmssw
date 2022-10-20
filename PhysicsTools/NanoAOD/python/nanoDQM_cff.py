import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

## Modify plots accordingly to era
_tauPlotsPreV9 = cms.VPSet()
for plot in nanoDQM.vplots.Tau.plots:
    if plot.name.value()!="idDecayModeOldDMs":
        _tauPlotsPreV9.append(plot)
_tauPlotsPreV9.extend([
                Plot1D('idDecayMode', 'idDecayMode', 2, -0.5, 1.5, "tauID('decayModeFinding')"),
                Plot1D('idDecayModeNewDMs', 'idDecayModeNewDMs', 2, -0.5, 1.5, "tauID('decayModeFindingNewDMs')"),
                Plot1D('idAntiEle', 'idAntiEle', 11, -0.5, 10.5, 'Anti-electron MVA discriminator V6: int 1 = VLoose, 2 = Loose, 3 = Medium, 4 = Tight, 5 = VTight'),
                Plot1D('idAntiEle2018', 'idAntiEle2018', 11, -0.5, 10.5, 'Anti-electron MVA discriminator V6 (2018): int 1 = VLoose, 2 = Loose, 3 = Medium, 4 = Tight, 5 = VTight'),
                Plot1D('idMVAnewDM2017v2', 'idMVAnewDM2017v2', 11, -0.5, 10.5, 'IsolationMVArun2v1DBnewDMwLT ID working point (2017v2): int 1 = VVLoose, 2 = VLoose, 3 = Loose, 4 = Medium, 5 = Tight, 6 = VTight, 7 = VVTight'),
                Plot1D('idMVAoldDM', 'idMVAoldDM', 11, -0.5, 10.5, 'IsolationMVArun2v1DBoldDMwLT ID working point: int 1 = VLoose, 2 = Loose, 3 = Medium, 4 = Tight, 5 = VTight, 6 = VVTight'),
                Plot1D('idMVAoldDM2017v1', 'idMVAoldDM2017v1', 11, -0.5, 10.5, 'IsolationMVArun2v1DBoldDMwLT ID working point (2017v1): int 1 = VVLoose, 2 = VLoose, 3 = Loose, 4 = Medium, 5 = Tight, 6 = VTight, 7 = VVTight'),
                Plot1D('idMVAoldDM2017v2', 'idMVAoldDM2017v2', 11, -0.5, 10.5, 'IsolationMVArun2v1DBoldDMwLT ID working point (2017v2): int 1 = VVLoose, 2 = VLoose, 3 = Loose, 4 = Medium, 5 = Tight, 6 = VTight, 7 = VVTight'),
                Plot1D('idMVAoldDMdR032017v2', 'idMVAoldDMdR032017v2', 11, -0.5, 10.5, 'IsolationMVArun2v1DBdR03oldDMwLT ID working point (217v2): int 1 = VVLoose, 2 = VLoose, 3 = Loose, 4 = Medium, 5 = Tight, 6 = VTight, 7 = VVTight'),
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

(run2_nanoAOD_106Xv1).toModify(
    nanoDQM.vplots.Tau, plots = _tauPlotsPreV9
)

_boostedTauPlotsV10 = cms.VPSet()
for plot in nanoDQM.vplots.boostedTau.plots:
    _boostedTauPlotsV10.append(plot)
_boostedTauPlotsV10.extend([
    Plot1D('idMVAoldDMdR032017v2', 'idMVAoldDMdR032017v2', 11, -0.5, 10.5, 'IsolationMVArun2017v2DBoldDMdR0p3wLT ID working point (2017v2): int 1 = VVLoose, 2 = VLoose, 3 = Loose, 4 = Medium, 5 = Tight, 6 = VTight, 7 = VVTight'),
    Plot1D('rawMVAoldDMdR032017v2', 'rawMVAoldDMdR032017v2', 20, -1, 1, 'byIsolationMVArun2017v2DBoldDMdR0p3wLT raw output discriminator (2017v2)')
])

(run2_nanoAOD_106Xv2).toModify(
    nanoDQM.vplots.boostedTau,
    plots = _boostedTauPlotsV10
)

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

(run2_nanoAOD_106Xv1).toModify(
    nanoDQM.vplots, LowPtElectron = None
).toModify(
    nanoDQMMC.vplots,
    LowPtElectron = None
)

nanoHarvest = cms.Sequence( nanoDQMQTester )
