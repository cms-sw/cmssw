import FWCore.ParameterSet.Config as cms

analyzePatJets = cms.EDAnalyzer("PatJetAnalyzer",
  src  = cms.InputTag("cleanPatJets"),                              
  corrLevel = cms.string("L3Absolute")
)                               
# foo bar baz
