import FWCore.ParameterSet.Config as cms

# Magnetic Field
# Geometries
# from Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
EstimatorForSTA = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'Chi2STA',
    MaxChi2 = 1000.
)
import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
KFTrajectoryFitterForSTA = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'KFFitterSTA',
    Propagator = 'SteppingHelixPropagatorAny',
    Estimator = 'Chi2STA'
)
import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
KFTrajectorySmootherForSTA = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName = 'KFSmootherSTA',
    Propagator = 'SteppingHelixPropagatorOpposite',
    Estimator = 'Chi2STA'
)
import TrackingTools.TrackFitters.KFFittingSmoother_cfi
KFFittingSmootheForSTA = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = 'KFFitterSmootherSTA',
    Fitter = 'KFFitterSTA',
    Smoother = 'KFSmootherSTA'
)
# Stand Alone Muons Producer
from RecoMuon.L2MuonProducer.L2Muons_cfi import *
