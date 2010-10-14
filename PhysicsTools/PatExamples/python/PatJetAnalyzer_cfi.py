import FWCore.ParameterSet.Config as cms

analyzePatJets = cms.EDAnalyzer("PatJetAnalyzer",
  ## input for reco jets
  reco = cms.InputTag("ak5CaloJets"),
  ## input for pat jets 
  src  = cms.InputTag("cleanPatJets"),                              
  ## correction level for pat jet in
  ## the format corrType:flavorType
  corrLevel = cms.string("abs")
)                               

analyzeJES = cms.EDAnalyzer("PatJetAnalyzer",
  ## input for pat jets 
  src  = cms.InputTag("cleanPatJets"),                              
  ## correction level for pat jet in
  ## the format corrType:flavorType
  corrLevel = cms.string("abs")
)                               
