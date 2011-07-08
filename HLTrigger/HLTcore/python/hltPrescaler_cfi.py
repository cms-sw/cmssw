import FWCore.ParameterSet.Config as cms

hltPrescaler = cms.EDFilter("HLTPrescaler",
  offset = cms.uint32(0),
  L1GtReadoutRecordTag = cms.InputTag("hltGtDigis")
)
