import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmoother_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
egammaCTFFinalFitWithMaterial = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
egammaCTFFinalFitWithMaterial.src = 'siStripElectrons'
egammaCTFFinalFitWithMaterial.Fitter = 'KFFittingSmoother'
egammaCTFFinalFitWithMaterial.Propagator = 'PropagatorWithMaterial'
egammaCTFFinalFitWithMaterial.alias = 'egammaCTFWithMaterialTracks'
egammaCTFFinalFitWithMaterial.TTRHBuilder = 'WithTrackAngle'
egammaCTFFinalFitWithMaterial.TrajectoryInEvent = False

