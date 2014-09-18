import FWCore.ParameterSet.Config as cms

Phase2TrackerHeaderProducer = cms.EDProducer(
    'Phase2TrackerHeaderProducer',
    ProductLabel = cms.InputTag("rawDataCollector")
)
