import FWCore.ParameterSet.Config as cms

Phase2TrackerDigitestproducer = cms.EDProducer(
    'Phase2TrackerDigi_test_producer',
    ProductLabel = cms.InputTag("source")
)
