import FWCore.ParameterSet.Config as cms

# Magnetic Field
# Geometries
# from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
EstimatorForSTA = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone()
import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
KFTrajectoryFitterForSTA = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone()
import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
KFTrajectorySmootherForSTA = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone()
import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
KFFittingSmootheForSTA = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
# Stand Alone Muons Producer
from RecoMuon.L2MuonProducer.L2Muons_cfi import *
EstimatorForSTA.ComponentName = 'Chi2STA'
EstimatorForSTA.MaxChi2 = 1000.
KFTrajectoryFitterForSTA.ComponentName = 'KFFitterSTA'
KFTrajectoryFitterForSTA.Propagator = 'SteppingHelixPropagatorAny'
KFTrajectoryFitterForSTA.Estimator = 'Chi2STA'
KFTrajectorySmootherForSTA.ComponentName = 'KFSmootherSTA'
KFTrajectorySmootherForSTA.Propagator = 'SteppingHelixPropagatorOpposite'
KFTrajectorySmootherForSTA.Estimator = 'Chi2STA'
KFFittingSmootheForSTA.ComponentName = 'KFFitterSmootherSTA'
KFFittingSmootheForSTA.Fitter = 'KFFitterSTA'
KFFittingSmootheForSTA.Smoother = 'KFSmootherSTA'


