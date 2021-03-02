import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from DQMOffline.Muon.gemEfficiencyAnalyzerCosmicsDefault_cfi import gemEfficiencyAnalyzerCosmicsDefault as _gemEfficiencyAnalyzerCosmicsDefault

gemEfficiencyAnalyzerCosmics = _gemEfficiencyAnalyzerCosmicsDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone(),
    muonTag = cms.InputTag('muons'),
    name = cms.untracked.string('Cosmic 2-Leg STA Muon'),
    folder = cms.untracked.string('GEM/Efficiency/type1'),
)

gemEfficiencyAnalyzerCosmicsOneLeg = _gemEfficiencyAnalyzerCosmicsDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone(),
    muonTag = cms.InputTag('muons1Leg'),
    name = cms.untracked.string('Cosmic 1-Leg STA Muon'),
    folder = cms.untracked.string('GEM/Efficiency/type2'),
)

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(gemEfficiencyAnalyzerCosmics,
    etaNbins=cms.untracked.int32(15),
    etaUp=cms.untracked.double(3.0))

phase2_GEM.toModify(gemEfficiencyAnalyzerCosmicsOneLeg,
    etaNbins=cms.untracked.int32(15),
    etaUp=cms.untracked.double(3.0))
