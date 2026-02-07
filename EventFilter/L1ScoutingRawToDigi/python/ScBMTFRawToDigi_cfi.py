import FWCore.ParameterSet.Config as cms

ScBMTFUnpacker = cms.EDProducer('ScBMTFRawToDigi',
  srcInputTag = cms.InputTag('rawDataCollector'),
  sourceIdList = cms.vint32(10,11,12,13,14,15,16,17,18,19,20,21)
)

