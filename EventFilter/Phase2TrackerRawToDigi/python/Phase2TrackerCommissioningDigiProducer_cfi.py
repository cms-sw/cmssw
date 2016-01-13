import FWCore.ParameterSet.Config as cms

Phase2TrackerCommissioningDigiProducer = cms.EDProducer(
    'Phase2TrackerCommissioningDigiProducer',
    ProductLabel = cms.InputTag("rawDataCollector")
)
