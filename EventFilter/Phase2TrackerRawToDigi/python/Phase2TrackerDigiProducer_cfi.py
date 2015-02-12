import FWCore.ParameterSet.Config as cms

Phase2TrackerDigiProducer = cms.EDProducer(
    'Phase2TrackerDigiProducer',
    ProductLabel = cms.InputTag("rawDataCollector")
)
