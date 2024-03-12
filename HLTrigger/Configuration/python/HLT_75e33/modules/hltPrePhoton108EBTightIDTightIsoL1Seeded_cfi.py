import FWCore.ParameterSet.Config as cms

hltPrePhoton108EBTightIDTightIsoL1Seeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
# foo bar baz
# PNCPH621Fupth
