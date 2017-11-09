import FWCore.ParameterSet.Config as cms

muonME0Digis = cms.EDProducer("ME0RawToDigiModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    # no DB mapping yet
    useDBEMap = cms.bool(False),    
)
