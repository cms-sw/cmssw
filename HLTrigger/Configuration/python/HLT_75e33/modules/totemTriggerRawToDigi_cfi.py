import FWCore.ParameterSet.Config as cms

totemTriggerRawToDigi = cms.EDProducer("TotemTriggerRawToDigi",
    fedId = cms.uint32(0),
    rawDataTag = cms.InputTag("rawDataCollector")
)
