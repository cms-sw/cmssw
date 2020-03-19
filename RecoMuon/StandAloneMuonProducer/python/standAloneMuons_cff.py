import FWCore.ParameterSet.Config as cms

# Magnetic Field
# Geometries
# from Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
#
#
#
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
EstimatorForSTA = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone()

EstimatorForSTA.ComponentName = 'Chi2STA'
EstimatorForSTA.MaxChi2 = 1000.
#
#
import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
KFTrajectoryFitterForSTA = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone()

KFTrajectoryFitterForSTA.ComponentName = 'KFFitterSTA'
KFTrajectoryFitterForSTA.Propagator = 'SteppingHelixPropagatorAny'
KFTrajectoryFitterForSTA.Estimator = 'Chi2STA'
#
#
import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
KFTrajectorySmootherForSTA = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone()

KFTrajectorySmootherForSTA.ComponentName = 'KFSmootherSTA'
KFTrajectorySmootherForSTA.Propagator = 'SteppingHelixPropagatorOpposite'
KFTrajectorySmootherForSTA.Estimator = 'Chi2STA'
#
#
import TrackingTools.TrackFitters.KFFittingSmoother_cfi
KFFittingSmootheForSTA = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone()

KFFittingSmootheForSTA.ComponentName = 'KFFitterSmootherSTA'
KFFittingSmootheForSTA.Fitter = 'KFFitterSTA'
KFFittingSmootheForSTA.Smoother = 'KFSmootherSTA'
#
#
# Stand Alone Muons Producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi import *
