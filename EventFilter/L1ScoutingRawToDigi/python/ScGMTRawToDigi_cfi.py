import FWCore.ParameterSet.Config as cms

ScGmtUnpacker = cms.EDProducer('ScGMTRawToDigi',
  srcInputTag = cms.InputTag('rawDataCollector'),
  # skip intermediate muons
  skipInterm = cms.bool(True),
  # print all objects
  debug = cms.untracked.bool(False)
)
