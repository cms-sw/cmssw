import FWCore.ParameterSet.Config as cms

hltPreDiphoton3023IsoCaloIdUnseeded = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("hltGtStage2Digis"),
    offset = cms.uint32(0)
)
# foo bar baz
# UVvpn2KvKKZqi
# EUbLfLzsCs9il
