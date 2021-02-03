import FWCore.ParameterSet.Config as cms

MTDTransientTrackingRecHitBuilder = cms.ESProducer("MTDTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('MTDRecHitBuilder')
)
