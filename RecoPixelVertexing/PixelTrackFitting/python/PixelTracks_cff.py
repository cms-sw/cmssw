import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
myTTRHBuilderWithoutAngle = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE = 'Fake',
    ComponentName = 'PixelTTRHBuilderWithoutAngle'
)
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import pixelFitterByHelixProjections
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByRiemannParaboloid_cfi import pixelFitterByRiemannParaboloid
from RecoPixelVertexing.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics
from RecoPixelVertexing.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import pixelTrackCleanerBySharedHits
from RecoPixelVertexing.PixelTrackFitting.pixelTracks_cfi import pixelTracks as _pixelTracks
from RecoTracker.TkTrackingRegions.globalTrackingRegion_cfi import globalTrackingRegion as _globalTrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU

# SEEDING LAYERS
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSeedLayers, initialStepHitDoublets, _initialStepCAHitQuadruplets

# TrackingRegion
pixelTracksTrackingRegions = _globalTrackingRegion.clone()
trackingLowPU.toReplaceWith(pixelTracksTrackingRegions, _globalTrackingRegionFromBeamSpot.clone())


# Pixel Quadruplets Tracking
pixelTracksSeedLayers = initialStepSeedLayers.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting")
)

pixelTracksHitDoublets = initialStepHitDoublets.clone(
    clusterCheck = "",
    seedingLayers = "pixelTracksSeedLayers",
    trackingRegions = "pixelTracksTrackingRegions"
)

pixelTracksHitQuadruplets = _initialStepCAHitQuadruplets.clone(
    doublets = "pixelTracksHitDoublets",
    SeedComparitorPSet = dict(clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting')
)

from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletHeterogeneousEDProducer_cfi import caHitQuadrupletHeterogeneousEDProducer as _caHitQuadrupletHeterogeneousEDProducer
gpu.toReplaceWith(pixelTracksHitQuadruplets, _caHitQuadrupletHeterogeneousEDProducer)
gpu.toModify(pixelTracksHitQuadruplets, trackingRegions = "pixelTracksTrackingRegions")

# for trackingLowPU
pixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "pixelTracksHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(
        clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"
    )
)

pixelTracks = _pixelTracks.clone(
    SeedingHitSets = "pixelTracksHitQuadruplets"
)
trackingLowPU.toModify(pixelTracks, SeedingHitSets = "pixelTracksHitTriplets")

from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromCUDA_cfi import pixelTrackProducerFromCUDA as _pixelTrackProducerFromCUDA
gpu.toReplaceWith(pixelTracks, _pixelTrackProducerFromCUDA)

pixelTracksTask = cms.Task(
    pixelTracksTrackingRegions,
    pixelFitterByHelixProjections,
    pixelTrackFilterByKinematics,
    pixelTracksSeedLayers,
    pixelTracksHitDoublets,
    pixelTracksHitQuadruplets,
    pixelTracks
)
_pixelTracksTask_lowPU = pixelTracksTask.copy()
_pixelTracksTask_lowPU.replace(pixelTracksHitQuadruplets, pixelTracksHitTriplets)
trackingLowPU.toReplaceWith(pixelTracksTask, _pixelTracksTask_lowPU)

# Use Riemann fit and substitute previous Fitter producer with the Riemann one
from Configuration.ProcessModifiers.riemannFit_cff import riemannFit
from Configuration.ProcessModifiers.riemannFitGPU_cff import riemannFitGPU
riemannFit.toModify(pixelTracks, Fitter = "pixelFitterByRiemannParaboloid")
riemannFitGPU.toModify(pixelTracks, runOnGPU = True)
_pixelTracksTask_riemannFit = pixelTracksTask.copy()
_pixelTracksTask_riemannFit.replace(pixelFitterByHelixProjections, pixelFitterByRiemannParaboloid)
riemannFit.toReplaceWith(pixelTracksTask, _pixelTracksTask_riemannFit)

pixelTracksSequence = cms.Sequence(pixelTracksTask)
