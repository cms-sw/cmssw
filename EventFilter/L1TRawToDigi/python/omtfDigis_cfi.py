import FWCore.ParameterSet.Config as cms

omtfDigis = cms.EDProducer("OmtfUnpacker",
  InputLabel = cms.InputTag('rawDataCollector'),
  useRpcConnectionFile = cms.bool(True)
)

