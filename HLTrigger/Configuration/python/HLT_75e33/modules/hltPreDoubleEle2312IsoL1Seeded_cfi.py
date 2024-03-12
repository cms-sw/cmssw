import FWCore.ParameterSet.Config as cms

hltPreDoubleEle2312IsoL1Seeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
# foo bar baz
# qtc7w9nUF4mAb
