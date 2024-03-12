import FWCore.ParameterSet.Config as cms

hltPreEle32WPTightL1Seeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
# foo bar baz
# fVAIapAG8Dcwf
# c1UmEuQFB6mPl
