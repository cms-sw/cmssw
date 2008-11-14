import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from RecoMuon.L3TrackFinder.MuonCkfTrajectoryBuilderESProducer_cff import *
from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
from FastSimulation.Muons.L3Muons_cfi import *
Chi2EstimatorForL3Refit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForL3Refit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

L3MuKFFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('L3MuKFFitter'),
    Estimator = cms.string('Chi2EstimatorForL3Refit'),
    Propagator = cms.string('SmartPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

L3MuKFSmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('L3MuKFSmoother'),
    errorRescaling = cms.double(100.0),
    Estimator = cms.string('Chi2EstimatorForL3Refit'),
    Propagator = cms.string('SmartPropagatorOpposite'),
    Updator = cms.string('KFUpdator')
)

# No use of Runge-Kutta propagator for tracks
SmartPropagatorAny.TrackerPropagator = 'PropagatorWithMaterial'

