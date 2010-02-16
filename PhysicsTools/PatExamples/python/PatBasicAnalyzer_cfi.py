import FWCore.ParameterSet.Config as cms

analyzeBasicPat = cms.EDAnalyzer("PatBasicAnalyzer",
  photonSrc   = cms.untracked.InputTag("cleanLayer1Photons"),
  electronSrc = cms.untracked.InputTag("cleanLayer1Electrons"),
  muonSrc     = cms.untracked.InputTag("cleanLayer1Muons"),                                             
  tauSrc      = cms.untracked.InputTag("cleanLayer1Taus"),
  jetSrc      = cms.untracked.InputTag("cleanLayer1Jets"),
  metSrc      = cms.untracked.InputTag("layer1METs")
)
