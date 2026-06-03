import FWCore.ParameterSet.Config as cms

hltMTDTransientTrackingRecHitBuilder = cms.ESProducer("MTDTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('hltMTDRecHitBuilder')
)
