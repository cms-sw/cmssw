import FWCore.ParameterSet.Config as cms

CloseComponentsMerger5D = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('CloseComponentsMerger5D'),
    DistanceMeasure = cms.string('KullbackLeiblerDistance5D'),
    MaxComponents = cms.int32(12)
)
