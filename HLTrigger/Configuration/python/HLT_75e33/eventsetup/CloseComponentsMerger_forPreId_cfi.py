import FWCore.ParameterSet.Config as cms

CloseComponentsMerger_forPreId = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('CloseComponentsMerger_forPreId'),
    DistanceMeasure = cms.string('KullbackLeiblerDistance5D'),
    MaxComponents = cms.int32(4)
)
