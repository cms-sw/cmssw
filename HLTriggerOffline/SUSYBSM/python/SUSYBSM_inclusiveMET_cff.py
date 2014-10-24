import FWCore.ParameterSet.Config as cms

SUSY_HLT_InclusiveMET = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET170_NoiseCleaned_v1'),
  TriggerFilter = cms.InputTag('hltPFMET170Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

