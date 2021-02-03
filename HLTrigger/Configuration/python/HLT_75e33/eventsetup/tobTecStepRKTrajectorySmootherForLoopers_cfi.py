import FWCore.ParameterSet.Config as cms

tobTecStepRKTrajectorySmootherForLoopers = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('tobTecStepRKSmootherForLoopers'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(7)
)
