import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

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

_Electron_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.Electron.plots:
    if 'Fall17V2' not in plot.name.value():
        _Electron_Run2_plots.append(plot)
_Electron_Run2_plots.extend([
    Plot1D('dEscaleUp', 'dEscaleUp', 100, -0.01, 0.01, '#Delta E scaleUp'),
    Plot1D('dEscaleDown', 'dEscaleDown', 100, -0.01, 0.01, '#Delta E scaleDown'),
    Plot1D('dEsigmaUp', 'dEsigmaUp', 100, -0.1, 0.1, '#Delta E sigmaUp'),
    Plot1D('dEsigmaDown', 'dEsigmaDown', 100, -0.1, 0.1, '#Delta E sigmaDown'),
    Plot1D('eCorr', 'eCorr', 20, 0.8, 1.2, 'ratio of the calibrated energy/miniaod energy'),
])
run2_egamma.toModify(
     nanoDQM.vplots.Electron, 
     plots = _Electron_Run2_plots
)

_Photon_Run2_plots = cms.VPSet()
def _match(name):
    if 'Fall17V2' in name: return True
    if '_quadratic' in name: return True
    if 'hoe_PUcorr' in name: return True
    return False
for plot in nanoDQM.vplots.Photon.plots:
    if not _match(plot.name.value()):
        _Photon_Run2_plots.append(plot)
_Photon_Run2_plots.extend([
    Plot1D('pfRelIso03_all', 'pfRelIso03_all', 20, 0, 2, 'PF relative isolation dR=0.3, total (with rho*EA PU Fall17V2 corrections)'),
    Plot1D('pfRelIso03_chg', 'pfRelIso03_chg', 20, 0, 2, 'PF relative isolation dR=0.3, charged component (with rho*EA PU Fall17V2 corrections)'),
    Plot1D('dEscaleUp', 'dEscaleUp', 100, -0.01, 0.01, '#Delta E scaleUp'),
    Plot1D('dEscaleDown', 'dEscaleDown', 100, -0.01, 0.01, '#Delta E scaleDown'),
    Plot1D('dEsigmaUp', 'dEsigmaUp', 100, -0.1, 0.1, '#Delta E sigmaUp'),
    Plot1D('dEsigmaDown', 'dEsigmaDown', 100, -0.1, 0.1, '#Delta E sigmaDown'),
    Plot1D('eCorr', 'eCorr', 20, 0.8, 1.2, 'ratio of the calibrated energy/miniaod energy'),
])
run2_egamma.toModify(
     nanoDQM.vplots.Photon, 
     plots = _Photon_Run2_plots
)

_FatJet_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.FatJet.plots:
    _FatJet_Run2_plots.append(plot)
_FatJet_Run2_plots.extend([
    Plot1D('btagCSVV2', 'btagCSVV2', 20, -1, 1, ' pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)'),
])

_Jet_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.Jet.plots:
    _Jet_Run2_plots.append(plot)
_Jet_Run2_plots.extend([
    Plot1D('btagCSVV2', 'btagCSVV2', 20, -1, 1, ' pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)'),
    Plot1D('btagCMVA', 'btagCMVA', 20, -1, 1, 'CMVA V2 btag discriminator'),
    Plot1D('btagDeepB', 'btagDeepB', 20, -1, 1, 'Deep B+BB btag discriminator'),
    Plot1D('btagDeepC', 'btagDeepC', 20, 0, 1, 'DeepCSV charm btag discriminator'),
    Plot1D('btagDeepCvB', 'btagDeepCvB', 20, -1, 1, 'DeepCSV c vs b+bb discriminator'),
    Plot1D('btagDeepCvL', 'btagDeepCvL', 20, -1, 1, 'DeepCSV c vs udsg discriminator')
])

_SubJet_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.SubJet.plots:
    _SubJet_Run2_plots.append(plot)
_SubJet_Run2_plots.extend([
    Plot1D('btagCSVV2', 'btagCSVV2', 20, -1, 1, ' pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)'),
])

run2_nanoAOD_ANY.toModify(
    nanoDQM.vplots.FatJet,
    plots = _FatJet_Run2_plots
).toModify(
    nanoDQM.vplots.Jet,
    plots = _Jet_Run2_plots
).toModify(
    nanoDQM.vplots.SubJet,
    plots = _SubJet_Run2_plots
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

nanoHarvest = cms.Sequence( nanoDQMQTester )
