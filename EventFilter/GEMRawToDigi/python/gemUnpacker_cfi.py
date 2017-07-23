import FWCore.ParameterSet.Config as cms

gemunpacker = cms.EDProducer("GEMUnpackingModule",
    InputLabel = cms.InputTag("rawDataCollector")
)


