import FWCore.ParameterSet.Config as cms

MTDTransientTrackingRecHitBuilderESProducer = cms.ESProducer("MTDTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('MTDRecHitBuilder')
)



