import FWCore.ParameterSet.Config as cms


l1MetFilterRecoTree = cms.EDAnalyzer("L1MetFilterRecoTreeProducer",
  triggerResultsToken               = cms.untracked.InputTag("TriggerResults"),
  hbheNoiseFilterResultToken        = cms.untracked.InputTag("HBHENoiseFilterResultProducer:HBHENoiseFilterResult")
 )


