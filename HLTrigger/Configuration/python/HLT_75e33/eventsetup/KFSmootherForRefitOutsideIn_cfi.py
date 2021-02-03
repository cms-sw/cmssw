import FWCore.ParameterSet.Config as cms

KFSmootherForRefitOutsideIn = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('KFSmootherForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3)
)
