import FWCore.ParameterSet.Config as cms

SUSY_HLT_MET_BTAG = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLTX'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  caloMETCollection = cms.InputTag("met"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLTX'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'),
  TriggerFilter = cms.InputTag('hltPFMET120Filter', '', 'HLTX'), #the last filter in the path, to use with test sample
  #TriggerFilter = cms.InputTag('hltPFHT350', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

