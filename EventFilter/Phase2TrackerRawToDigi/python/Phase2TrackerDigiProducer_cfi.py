import FWCore.ParameterSet.Config as cms

Phase2TrackerDigitestproducer = cms.EDProducer(
    'Phase2TrackerDigiProducer',
    ProductLabel = cms.InputTag("rawDataCollector")
)
