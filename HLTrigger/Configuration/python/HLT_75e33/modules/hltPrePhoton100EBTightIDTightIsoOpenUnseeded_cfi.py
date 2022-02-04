import FWCore.ParameterSet.Config as cms

hltPrePhoton100EBTightIDTightIsoOpenUnseeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
