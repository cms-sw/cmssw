import FWCore.ParameterSet.Config as cms

gemRawToDigi = cms.EDProducer("GEMRawToDigiModule",
    InputObjects = cms.InputTag("rawDataCollector"),
)
