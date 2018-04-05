import FWCore.ParameterSet.Config as cms

totemTriggerRawToDigi = cms.EDProducer("TotemTriggerRawToDigi",
  rawDataTag = cms.InputTag(""),

  # IMPORTANT: leave 0 to load the default configuration from
  #    DataFormats/FEDRawData/interface/FEDNumbering.h
  fedId = cms.uint32(0)
)
