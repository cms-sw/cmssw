import FWCore.ParameterSet.Config as cms

RKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('RKSmoother'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    errorRescaling = cms.double(100),
    minHits = cms.int32(3)
)
