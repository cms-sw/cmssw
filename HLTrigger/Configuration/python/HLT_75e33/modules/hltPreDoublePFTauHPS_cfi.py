
import FWCore.ParameterSet.Config as cms

hltPreDoublePFTauHPS = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
