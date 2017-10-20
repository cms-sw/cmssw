import FWCore.ParameterSet.Config as cms

gemDigis = cms.EDProducer("GEMRawToDigiModule",
    InputObjects = cms.InputTag("rawDataCollector"),
)
