import FWCore.ParameterSet.Config as cms

SUSY_HLT_InclusiveHT = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLTX'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  caloMETCollection = cms.InputTag("met"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLTX'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFHT900_v1'),
  TriggerFilter = cms.InputTag('hltPFHT900', '', 'HLTX'), #the last filter in the path, to use with test sample
  #TriggerFilter = cms.InputTag('hltPFHT900', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

