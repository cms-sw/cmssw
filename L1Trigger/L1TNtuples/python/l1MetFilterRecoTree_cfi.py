import FWCore.ParameterSet.Config as cms


l1MetFilterRecoTree = cms.EDAnalyzer("L1MetFilterRecoTreeProducer",
  triggerResultsToken               = cms.untracked.InputTag("TriggerResults::RECO"),
  hbheNoiseFilterResultToken        = cms.untracked.InputTag("HBHENoiseFilterResultProducer:HBHENoiseFilterResult"),
  badChargedCandidateFilterToken    = cms.untracked.InputTag("BadChargedCandidateFilter"),
  badPFMuonFilterToken              = cms.untracked.InputTag("BadPFMuonFilter")
 )


