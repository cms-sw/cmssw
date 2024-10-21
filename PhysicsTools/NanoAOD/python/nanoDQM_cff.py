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

## EGamma custom nano
_Electron_extra_plots = nanoDQM.vplots.Electron.plots.copy()
_Electron_extra_plots.extend([
    Plot1D('r9Frac', 'r9Frac', 22, 0, 1.1, 'Fractional R9'),
    Plot1D('DeltaEtaInSC', 'DeltaEtaInSC', 20, -0.01, 0.01, 'DEta trk and SuperClus'),
    Plot1D('DeltaEtaInSeed', 'DeltaEtaInSeed', 20, -0.005, 0.005, 'DEta trk and SeedClus'),
    Plot1D('DeltaPhiInSC', 'DeltaPhiInSC', 20, -0.01, 0.01, 'DPhi trk and SuperClus'),
    Plot1D('DeltaPhiInSeed', 'DeltaPhiInSeed', 20, -0.01, 0.01, 'DPhi trk and SeedClus'),
    Plot1D('full5x5HoverE', 'full5x5HoverE', 20, 0, 0.2, 'full5x5 H/E'),
    Plot1D('eSCOverP', 'eSCOverP', 20, 0, 2.0, 'supercluster E/p'),
    Plot1D('eEleOverPout', 'eEleOverPout', 20, 0, 2.0, 'supercluster E/pout'),
    Plot1D('e1x5', 'e1x5', 20, 0, 20, 'E1x5'),
    Plot1D('e2x5max', 'e2x5max', 20, 0, 20, 'E2x5Max'),
    Plot1D('e5x5', 'e5x5', 20, 0, 20, 'E5x5'),
    Plot1D('closestKFchi2', 'closestKFchi2', 20, 0, 20, 'closest KF trk chi2'),
    Plot1D('closestKFNLayers', 'closestKFNLayers', 20, 0, 19, 'closest KF NLayers'),
    Plot1D('dr03HcalTowerSumEt', 'dr03HcalTowerSumEt', 20, 0, 40, 'Hcal isolation'),
    Plot1D('GSFchi2', 'GSFchi2', 20, 0, 20, 'GSF chi2'),
    Plot1D('superclusterEta', 'superclusterEta', 30, 3.0, 3.0, 'supercluster Eta'),
    Plot1D('ecalPFClusIso', 'ecalPFClusIso', 20, 0, 40, 'ecal PF Cluster Isolation'),
    Plot1D('hcalPFClusIso', 'hcalPFClusIso', 20, 0, 40, 'hcal PF Cluster Isolation'),
    Plot1D('nBrem', 'nBrem', 20, 0, 19, 'num of Brem'),
    Plot1D('pfPhotonIso', 'pfPhotonIso', 20, 0, 20, 'pf Photon Isolation'),
    Plot1D('pfChargedHadIso', 'pfChargedHadIso', 20, 0, 40, 'pf Charged Hadron Isolation'),
    Plot1D('pfNeutralHadIso', 'pfNeutralHadIso', 20, 0, 40, 'pfNeutralHadIso'),
    Plot1D('sigmaIphiIphiFull5x5', 'sigmaIphiIphiFull5x5', 20, 0, 0.1, 'sigmaIphiIphi Full5x5'),
    Plot1D('etaWidth', 'etaWidth', 20, 0, 0.05, 'eta Width'),
    Plot1D('phiWidth', 'phiWidth', 20, 0, 0.1, 'phi Width'),
    Plot1D('seedClusEnergy', 'seedClusEnergy', 20, 0, 40, 'seedClusEnergy'),
    Plot1D('hoeSingleTower', 'hoeSingleTower', 20, 0, 0.2, 'Single Tower H/E'),
    Plot1D('hoeFull5x5', 'hoeFull5x5', 20, 0, 0.2, 'Full5x5 H/E'),
    Plot1D('sigmaIetaIphiFull5x5', 'sigmaIetaIphiFull5x5', 20, 0, 0.2, 'full5x5 sigmaIetaIphi'),
    Plot1D('eMax', 'eMax', 20, 0, 40, 'eMax'),
    Plot1D('e2nd', 'e2nd', 20, 0, 40, 'e2nd'),
    Plot1D('eTop', 'eTop', 20, 0, 40, 'eTop'),
    Plot1D('eBottom', 'eBottom', 20, 0, 40, 'eBottom'),
    Plot1D('eLeft', 'eLeft', 20, 0, 40, 'eLeft'),
    Plot1D('eRight', 'eRight', 20, 0, 40, 'eRight'),
    Plot1D('e2x5Top', 'e2x5Top', 20, 0, 40, 'e2x5Top'),
    Plot1D('e2x5Bottom', 'e2x5Bottom', 20, 0, 40, 'e2x5Bottom'),
    Plot1D('e2x5Left', 'e2x5Left', 20, 0, 40, 'e2x5Left'),
    Plot1D('e2x5Right', 'e2x5Right', 20, 0, 40, 'e2x5Right'),
    Plot1D('nSaturatedXtals', 'nSaturatedXtals', 10, 0, 9, 'number of Saturated crystals'),
    Plot1D('numberOfClusters', 'numberOfClusters', 10, 0, 9, 'number of Clusters'),
    Plot1D('istrackerDriven', 'istrackerDriven', 2, 0, 1, 'istrackerDriven'),
    Plot1D('superclusterPhi', 'superclusterPhi', 32, -3.2, 3.2, 'supercluster Phi'),
    Plot1D('seedClusterEta', 'seedClusterEta', 30, -3.0, 3.0, 'seedCluster Eta'),
    Plot1D('seedClusterPhi', 'seedClusterPhi', 32, -3.2, 3.2, 'seedCluster Phi'),
    Plot1D('superclusterEnergy', 'superclusterEnergy', 80, 0, 80, 'supercluster Energy'),
    Plot1D('energy', 'energy', 20, 0, 80, 'energy'),
    Plot1D('trackMomentumError', 'trackMomentumError', 20, 0, 1.0, 'trackMomentumError'),
    Plot1D('trackMomentum', 'trackMomentum', 20, 0, 80, 'trackMomentum'),
    Plot1D('trkLayersWithMeas', 'trkLayersWithMeas', 20, 0, 19, 'trkLayersWithMeas'),
    Plot1D('nValidPixBarrelHits', 'nValidPixBarrelHits', 5, 0, 4, 'nValidPixBarrelHits'),
    Plot1D('nValidPixEndcapHits', 'nValidPixEndcapHits', 20, 0, 19, 'nValidPixEndcapHits'),
    Plot1D('superClusterFbrem', 'superClusterFbrem', 12, 0, 1.2, 'superClusterFbrem'),
    Plot1D('convVtxFitProb', 'convVtxFitProb', 12, 0, 1.2, 'convVtxFitProb'),
    Plot1D('clustersSize', 'clustersSize', 20, 0, 19, 'clustersSize'),
    Plot1D('iEtaMod5', 'iEtaMod5', 20, 0, 40, 'iEtaMod5'),
    Plot1D('iEtaMod20', 'iEtaMod20', 20, 0, 40, 'iEtaMod20'),
    Plot1D('iPhiMod2', 'iPhiMod2', 20, 0, 199, 'iPhiMod2'),
    Plot1D('iPhiMod20', 'iPhiMod20', 100, 0, 99, 'iPhiMod20')
])

_Photon_extra_plots = nanoDQM.vplots.Photon.plots.copy()
_Photon_extra_plots.extend([
    Plot1D('r9Frac', 'r9Frac', 22, 0, 1.1, 'Fractional R9'),
    Plot1D('energy', 'energy', 20, 0, 80, 'energy'),
    Plot1D('rawPreshowerEnergy', 'rawPreshowerEnergy', 20, 0, 80, 'rawPreshowerEnergy'),
    Plot1D('seedClusEnergy', 'seedClusEnergy', 20, 0, 40, 'seedClusEnergy'),
    Plot1D('e5x5', 'e5x5', 20, 0, 20, 'E5x5'),
    Plot1D('dEtaSeedClusSuperClus', 'dEtaSeedClusSuperClus', 20, 0, 1.0, 'dEtaSeedClusSuperClus'),
    Plot1D('dPhiSeedClusSuperClus', 'dPhiSeedClusSuperClus', 20, 0, 1.0, 'dPhiSeedClusSuperClus'),
    Plot1D('sigmaIphiIphiFull5x5', 'sigmaIphiIphiFull5x5', 20, 0, 0.1, 'sigmaIphiIphi Full5x5'),
    Plot1D('eMax', 'eMax', 20, 0, 40, 'eMax'),
    Plot1D('e2nd', 'e2nd', 20, 0, 40, 'e2nd'),
    Plot1D('eTop', 'eTop', 20, 0, 40, 'eTop'),
    Plot1D('eBottom', 'eBottom', 20, 0, 40, 'eBottom'),
    Plot1D('eLeft', 'eLeft', 20, 0, 40, 'eLeft'),
    Plot1D('eRight', 'eRight', 20, 0, 40, 'eRight'),
    Plot1D('e2x5Top', 'e2x5Top', 20, 0, 40, 'e2x5Top'),
    Plot1D('e2x5Bottom', 'e2x5Bottom', 20, 0, 40, 'e2x5Bottom'),
    Plot1D('e2x5Left', 'e2x5Left', 20, 0, 40, 'e2x5Left'),
    Plot1D('e2x5Right', 'e2x5Right', 20, 0, 40, 'e2x5Right'),
    Plot1D('nSaturatedXtals', 'nSaturatedXtals', 10, 0, 9, 'number of Saturated crystals'),
    Plot1D('numberOfClusters', 'numberOfClusters', 10, 0, 9, 'number of Clusters'),
    Plot1D('hadTowOverEm', 'hadTowOverEm', 20, 0, 0.2, 'Single Tower H/E'),
    Plot1D('ecalRecHitIsolation', 'ecalRecHitIsolation', 20, 0, 40, 'ecal RecHit Isolation'),
    Plot1D('sigmaIetaIetaFrac', 'sigmaIetaIetaFrac', 20, 0, 0.08, 'sigmaIetaIetaFrac'),
    Plot1D('chargedHadronIso', 'chargedHadronIso', 20, 0, 40, 'chargedHadronIso'),
    Plot1D('iEtaMod5', 'iEtaMod5', 20, 0, 40, 'iEtaMod5'),
    Plot1D('iEtaMod20', 'iEtaMod20', 20, 0, 40, 'iEtaMod20'),
    Plot1D('iPhiMod2', 'iPhiMod2', 20, 0, 199, 'iPhiMod2'),
    Plot1D('iPhiMod20', 'iPhiMod20', 100, 0, 99, 'iPhiMod20')
])

_Electron_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.Electron.plots:
    if 'Fall17V2' not in plot.name.value():
        _Electron_Run2_plots.append(plot)
_Electron_Run2_plots.extend([
    Plot1D('dEscaleUp', 'dEscaleUp', 100, -0.01, 0.01, '#Delta E scaleUp'),
    Plot1D('dEscaleDown', 'dEscaleDown', 100, -0.01, 0.01, '#Delta E scaleDown'),
    Plot1D('dEsigmaUp', 'dEsigmaUp', 100, -0.1, 0.1, '#Delta E sigmaUp'),
    Plot1D('dEsigmaDown', 'dEsigmaDown', 100, -0.1, 0.1, '#Delta E sigmaDown'),
    Plot1D('ptPreCorr', 'ptPreCorr', 100, 0., 500., 'Pt before scale & smearing energy corrections'),
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
    Plot1D('ptPreCorr', 'ptPreCorr', 100, 0., 500., 'Pt before scale & smearing energy corrections'),
])
run2_egamma.toModify(
     nanoDQM.vplots.Photon,
     plots = _Photon_Run2_plots
)

_FatJet_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.FatJet.plots:
    if 'EF' not in plot.name.value():
        _FatJet_Run2_plots.append(plot)
_FatJet_Run2_plots.extend([
    Plot1D('btagCSVV2', 'btagCSVV2', 20, -1, 1, ' pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)'),
    Plot1D('btagDeepB', 'btagDeepB', 20, -1, 1, 'Deep B+BB btag discriminator'),
    Plot1D('btagHbb', 'btagHbb', 20, -1, 1, 'Higgs to BB tagger discriminator'),
    Plot1D('btagCMVA', 'btagCMVA', 20, -1, 1, 'CMVA V2 btag discriminator'),
    Plot1D('btagDDBvLV2', 'btagDDBvLV2', 20, 0, 1, 'DeepDoubleX V2(mass-decorrelated) discriminator for H(Z)->bb vs QCD'),
    Plot1D('btagDDCvBV2', 'btagDDCvBV2', 20, 0, 1, 'DeepDoubleX V2 (mass-decorrelated) discriminator for H(Z)->cc vs H(Z)->bb'),
    Plot1D('btagDDCvLV2', 'btagDDCvLV2', 20, 0, 1, 'DeepDoubleX V2 (mass-decorrelated) discriminator for H(Z)->cc vs QCD'),
    Plot1D('deepTagMD_H4qvsQCD', 'deepTagMD_H4qvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger H->4q vs QCD discriminator'),
    Plot1D('deepTagMD_HbbvsQCD', 'deepTagMD_HbbvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger H->bb vs QCD discriminator'),
    Plot1D('deepTagMD_TvsQCD', 'deepTagMD_TvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger top vs QCD discriminator'),
    Plot1D('deepTagMD_WvsQCD', 'deepTagMD_WvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger W vs QCD discriminator'),
    Plot1D('deepTagMD_ZHbbvsQCD', 'deepTagMD_ZHbbvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger Z/H->bb vs QCD discriminator'),
    Plot1D('deepTagMD_ZHccvsQCD', 'deepTagMD_ZHccvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger Z/H->cc vs QCD discriminator'),
    Plot1D('deepTagMD_ZbbvsQCD', 'deepTagMD_ZbbvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger Z->bb vs QCD discriminator'),
    Plot1D('deepTagMD_ZvsQCD', 'deepTagMD_ZvsQCD', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger Z vs QCD discriminator'),
    Plot1D('deepTagMD_bbvsLight', 'deepTagMD_bbvsLight', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger Z/H/gluon->bb vs light flavour discriminator'),
    Plot1D('deepTagMD_ccvsLight', 'deepTagMD_ccvsLight', 20, 0, 1, 'Mass-decorrelated DeepBoostedJet tagger Z/H/gluon->cc vs light flavour discriminator'),
    Plot1D('deepTag_H', 'deepTag_H', 20, 0, 1, 'DeepBoostedJet tagger H(bb,cc,4q) sum'),
    Plot1D('deepTag_QCD', 'deepTag_QCD', 20, 0, 1, 'DeepBoostedJet tagger QCD(bb,cc,b,c,others) sum'),
    Plot1D('deepTag_QCDothers', 'deepTag_QCDothers', 20, 0, 1, 'DeepBoostedJet tagger QCDothers value'),
    Plot1D('deepTag_TvsQCD', 'deepTag_TvsQCD', 20, 0, 1, 'DeepBoostedJet tagger top vs QCD discriminator'),
    Plot1D('deepTag_WvsQCD', 'deepTag_WvsQCD', 20, 0, 1, 'DeepBoostedJet tagger W vs QCD discriminator'),
    Plot1D('deepTag_ZvsQCD', 'deepTag_ZvsQCD', 20, 0, 1, 'DeepBoostedJet tagger Z vs QCD discriminator'),
    Plot1D('particleNetLegacy_mass', 'particleNetLegacy_mass', 25, 0, 250, 'ParticleNet Legacy Run-2 mass regression'),
    Plot1D('particleNetLegacy_Xbb', 'particleNetLegacy_Xbb', 20, 0, 1, 'ParticleNet Legacy Run-2 X->bb score'),
    Plot1D('particleNetLegacy_Xcc', 'particleNetLegacy_Xcc', 20, 0, 1, 'ParticleNet Legacy Run-2 X->cc score'),
    Plot1D('particleNetLegacy_Xqq', 'particleNetLegacy_Xqq', 20, 0, 1, 'ParticleNet Legacy Run-2 X->qq (uds) score'),
    Plot1D('particleNetLegacy_QCD', 'particleNetLegacy_QCD', 20, 0, 1, 'ParticleNet Legacy Run-2 QCD score'),
])

_Jet_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.Jet.plots:
    _Jet_Run2_plots.append(plot)
    if 'Multiplicity' not in plot.name.value() and 'hfHEF' not in plot.name.value() and 'hfEmEF' not in plot.name.value():
        _Jet_Run2_plots.append(plot)
_Jet_Run2_plots.extend([
    Plot1D('btagCSVV2', 'btagCSVV2', 20, -1, 1, ' pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)'),
    Plot1D('btagCMVA', 'btagCMVA', 20, -1, 1, 'CMVA V2 btag discriminator'),
    Plot1D('btagDeepB', 'btagDeepB', 20, -1, 1, 'Deep B+BB btag discriminator'),
    Plot1D('btagDeepC', 'btagDeepC', 20, 0, 1, 'DeepCSV charm btag discriminator'),
    Plot1D('btagDeepCvB', 'btagDeepCvB', 20, -1, 1, 'DeepCSV c vs b+bb discriminator'),
    Plot1D('btagDeepCvL', 'btagDeepCvL', 20, -1, 1, 'DeepCSV c vs udsg discriminator')
])
_Jet_pre142X_plots = cms.VPSet()
for plot in nanoDQM.vplots.Jet.plots:
    if 'puIdDisc' not in plot.name.value():
        _Jet_pre142X_plots.append(plot)

_SubJet_Run2_plots = cms.VPSet()
for plot in nanoDQM.vplots.SubJet.plots:
    _SubJet_Run2_plots.append(plot)
_SubJet_Run2_plots.extend([
    Plot1D('btagCSVV2', 'btagCSVV2', 20, -1, 1, ' pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)'),
])

_SubJet_pre142X_plots = cms.VPSet()
for plot in nanoDQM.vplots.SubJet.plots:
    if 'btagDeepFlavB' not in plot.name.value() and 'btagUParTAK4B' not in plot.name.value():
        _SubJet_pre142X_plots.append(plot)
_SubJet_pre142X_plots.extend([
    Plot1D('btagDeepB', 'btagDeepB', 20, -1, 1, 'Deep B+BB btag discriminator'),
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
run3_nanoAOD_pre142X.toModify(
    nanoDQM.vplots.Jet,
    plots = _Jet_pre142X_plots
).toModify(
    nanoDQM.vplots.SubJet,
    plots = _SubJet_pre142X_plots
)

_Pileup_pre13X_plots = cms.VPSet()
for plot in nanoDQM.vplots.Pileup.plots:
    if 'pthatmax' not in plot.name.value():
        _Pileup_pre13X_plots.append(plot)

(run2_nanoAOD_ANY).toModify(
    nanoDQM.vplots.Pileup,
    plots = _Pileup_pre13X_plots
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
