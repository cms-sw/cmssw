import FWCore.ParameterSet.Config as cms

omtfStage2Digis = cms.EDProducer("OmtfUnpacker",
  InputLabel = cms.InputTag('rawDataCollector'),
  outputTag = cms.string("OmtfUnpacker")
)

