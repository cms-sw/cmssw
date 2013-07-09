import FWCore.ParameterSet.Config as cms

analyzeBasicPat = cms.EDAnalyzer("PatBasicAnalyzer",
  photonSrc   = cms.untracked.InputTag("cleanPatPhotons"),
  electronSrc = cms.untracked.InputTag("cleanPatElectrons"),
  muonSrc     = cms.untracked.InputTag("cleanPatMuons"),                                             
  tauSrc      = cms.untracked.InputTag("cleanPatTaus"),
  jetSrc      = cms.untracked.InputTag("cleanPatJets"),
  metSrc      = cms.untracked.InputTag("patMETs")
)
