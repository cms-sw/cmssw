import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:patTuple.root'
  )
)

process.MessageLogger = cms.Service("MessageLogger")

process.analyzeBasicPat = cms.EDAnalyzer("WrappedEDMuonAnalyzer",
  muons     = cms.InputTag("cleanPatMuons"),                                             
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatBasics.root')
)

process.p = cms.Path(process.analyzeBasicPat)

