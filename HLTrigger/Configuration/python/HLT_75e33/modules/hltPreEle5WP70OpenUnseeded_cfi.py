import FWCore.ParameterSet.Config as cms

hltPreEle5WP70OpenUnseeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
