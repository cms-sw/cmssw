import FWCore.ParameterSet.Config as cms

omtfStage2Digis = cms.EDProducer("OmtfUnpacker",
    inputLabel = cms.InputTag("rawDataCollector"),
    skipRpc = cms.bool(False)
)
