import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
egammaCTFFinalFitWithMaterial = copy.deepcopy(ctfWithMaterialTracks)
egammaCTFFinalFitWithMaterial.src = 'siStripElectrons'
egammaCTFFinalFitWithMaterial.Fitter = 'KFFittingSmoother'
egammaCTFFinalFitWithMaterial.Propagator = 'PropagatorWithMaterial'
egammaCTFFinalFitWithMaterial.alias = 'egammaCTFWithMaterialTracks'
egammaCTFFinalFitWithMaterial.TTRHBuilder = 'WithTrackAngle'
egammaCTFFinalFitWithMaterial.TrajectoryInEvent = False

