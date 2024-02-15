import FWCore.ParameterSet.Config as cms

ScGmtUnpacker = cms.EDProducer('ScGMTRawToDigi',
  srcInputTag = cms.InputTag('rawDataCollector'),
  # print all objects
  debug = cms.untracked.bool(False)
)