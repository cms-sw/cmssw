import FWCore.ParameterSet.Config as cms

muonGEMDigis = cms.EDProducer("GEMRawToDigiModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    UnpackStatusDigis = cms.bool(False),
    useDBEMap = cms.bool(False),
)
