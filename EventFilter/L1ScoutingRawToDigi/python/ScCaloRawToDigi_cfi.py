import FWCore.ParameterSet.Config as cms

ScCaloUnpacker = cms.EDProducer("ScCaloRawToDigi",
  srcInputTag = cms.InputTag("rawDataCollector"),
  # print all objects
  debug = cms.untracked.bool(False)
)