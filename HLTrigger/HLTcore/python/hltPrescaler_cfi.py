import FWCore.ParameterSet.Config as cms

hltPrescaler = cms.EDFilter("HLTPrescaler",
  L1GtReadoutRecordTag = cms.InputTag("hltGtDigis")
)
