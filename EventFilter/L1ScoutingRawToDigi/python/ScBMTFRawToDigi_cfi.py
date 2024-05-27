import FWCore.ParameterSet.Config as cms

ScBMTFUnpacker = cms.EDProducer('ScBMTFRawToDigi',
  srcInputTag = cms.InputTag('rawDataCollector'),
  # print all objects
  debug = cms.untracked.bool(False)
)

