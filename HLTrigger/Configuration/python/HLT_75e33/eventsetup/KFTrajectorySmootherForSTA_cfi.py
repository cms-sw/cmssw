import FWCore.ParameterSet.Config as cms

KFTrajectorySmootherForSTA = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('KFSmootherSTA'),
    Estimator = cms.string('Chi2STA'),
    Propagator = cms.string('SteppingHelixPropagatorOpposite'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    errorRescaling = cms.double(100),
    minHits = cms.int32(3)
)
