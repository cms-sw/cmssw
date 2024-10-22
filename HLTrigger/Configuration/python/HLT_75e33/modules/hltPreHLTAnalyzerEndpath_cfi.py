import FWCore.ParameterSet.Config as cms

hltPreHLTAnalyzerEndpath = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
