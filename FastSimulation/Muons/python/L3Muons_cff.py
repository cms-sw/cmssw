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

HLTKFFittingSmoother = cms.ESProducer(
    "KFFittingSmootherESProducer",
    ComponentName = cms.string( "HLTKFFittingSmoother" ),
    Fitter = cms.string( "HLTKFFitter" ),
    Smoother = cms.string( "HLTKFSmoother" ),
    EstimateCut = cms.double( -1.0 ),
    MinNumberOfHits = cms.int32( 5 ),
    RejectTracks = cms.bool( True ),
    BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
    NoInvalidHitsBeginEnd = cms.bool( False ),
    appendToDataLabel = cms.string( "" )
)

HLTKFFitter = cms.ESProducer(
    "KFTrajectoryFitterESProducer",
    ComponentName = cms.string( "HLTKFFitter" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    Updator = cms.string( "KFUpdator" ),
    Estimator = cms.string( "Chi2" ),
    minHits = cms.int32( 3 ),
    appendToDataLabel = cms.string( "" )
)

HLTKFSmoother = cms.ESProducer(
    "KFTrajectorySmootherESProducer",
    ComponentName = cms.string( "HLTKFSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    Updator = cms.string( "KFUpdator" ),
    Estimator = cms.string( "Chi2" ),
    errorRescaling = cms.double( 100.0 ),
    minHits = cms.int32( 3 ),
    appendToDataLabel = cms.string( "" )
)


# No use of Runge-Kutta propagator for tracks
SmartPropagatorAny.TrackerPropagator = 'PropagatorWithMaterial'

