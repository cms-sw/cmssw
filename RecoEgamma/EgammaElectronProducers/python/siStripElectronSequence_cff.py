import FWCore.ParameterSet.Config as cms

#
#
# complete sequence to 1) make siStripElectrons
#                      2) do tracking based on these siStripElectrons
#                      3) associate tracks to SiStripElectrons
#
# Created by Shahram Rahatlou, University of Rome & INFN, 4 Aug 2006
# based on the cfg files from Jim Pivarsky, Cornell
#
# tracker geometry
# tracker numbering
# standard geometry
# magnetic field
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
# use super clusters and si strip hits to make siStripElectrons
from RecoEgamma.EgammaElectronProducers.siStripElectrons_cfi import *
# asscoiate tracks to siStripElectrons
from RecoEgamma.EgammaElectronProducers.siStripElectronToTrackAssociator_cfi import *

# do tracking seeded by siStripElectrons

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

siStripElectronSequence = cms.Sequence(siStripElectrons*egammaCTFFinalFitWithMaterial*siStripElectronToTrackAssociator)
