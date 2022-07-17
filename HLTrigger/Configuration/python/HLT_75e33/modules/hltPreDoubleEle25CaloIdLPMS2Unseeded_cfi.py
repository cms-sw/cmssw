import FWCore.ParameterSet.Config as cms

hltPreDoubleEle25CaloIdLPMS2Unseeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
