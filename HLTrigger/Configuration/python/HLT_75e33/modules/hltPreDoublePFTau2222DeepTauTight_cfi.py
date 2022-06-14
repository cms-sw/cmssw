import FWCore.ParameterSet.Config as cms

hltPreDoublePFTau2222DeepTauTight = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
