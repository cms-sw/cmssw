import FWCore.ParameterSet.Config as cms

Phase2TrackerDigiCondDataproducer = cms.EDProducer(
    'Phase2TrackerDigi_CondData_producer',
    ProductLabel = cms.InputTag("source")
)
