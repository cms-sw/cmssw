import FWCore.ParameterSet.Config as cms

hltPreDoubleEle25CaloIdLPMS2L1Seeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
