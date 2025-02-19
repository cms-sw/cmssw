import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:patTuple.root'
  )
)

process.MessageLogger = cms.Service("MessageLogger")

process.analyzeBasicPat = cms.EDAnalyzer("PatBasicAnalyzer",
  photonSrc   = cms.untracked.InputTag("cleanPatPhotons"),
  electronSrc = cms.untracked.InputTag("cleanPatElectrons"),
  muonSrc     = cms.untracked.InputTag("cleanPatMuons"),                                             
  tauSrc      = cms.untracked.InputTag("cleanPatTaus"),
  jetSrc      = cms.untracked.InputTag("cleanPatJets"),
  metSrc      = cms.untracked.InputTag("patMETs")
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatBasics.root')
)

process.p = cms.Path(process.analyzeBasicPat)

