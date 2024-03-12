import FWCore.ParameterSet.Config as cms

ScGmtUnpacker = cms.EDProducer('ScGMTRawToDigi',
  srcInputTag = cms.InputTag('rawDataCollector'),
  # print all objects
  debug = cms.untracked.bool(False)
)# foo bar baz
# 77xLvnVUxWBt5
# Jj7ZyO6eTVwXm
