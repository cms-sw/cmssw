import FWCore.ParameterSet.Config as cms

hltPreDiphoton3023IsoCaloIdL1Seeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
# foo bar baz
# LlxWkUjvjD1nk
# wyH1V86rcp68U
