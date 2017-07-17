import FWCore.ParameterSet.Config as cms

MaskedMeasurementTrackerEvent = cms.EDProducer("MaskedMeasurementTrackerEventProducer",
    src = cms.InputTag("MeasurementTrackerEvent"),
    OnDemand = cms.bool(False),
    clustersToSkip = cms.InputTag(""),
)
