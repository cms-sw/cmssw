import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

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
from RecoPixelVertexing.PixelTrackFitting.pixelNtupletsFitter_cfi import pixelNtupletsFitter
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

# Eras
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
from Configuration.Eras.Modifier_run3_common_cff import run3_common

# seeding layers
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSeedLayers, initialStepHitDoublets, _initialStepCAHitQuadruplets

# TrackingRegion
pixelTracksTrackingRegions = _globalTrackingRegion.clone()
trackingLowPU.toReplaceWith(pixelTracksTrackingRegions, _globalTrackingRegionFromBeamSpot.clone())


# Pixel quadruplets tracking
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

pixelTracks = _pixelTracks.clone(
    SeedingHitSets = "pixelTracksHitQuadruplets"
)

pixelTracksTask = cms.Task(
    pixelTracksTrackingRegions,
    pixelFitterByHelixProjections,
    pixelTrackFilterByKinematics,
    pixelTracksSeedLayers,
    pixelTracksHitDoublets,
    pixelTracksHitQuadruplets,
    pixelTracks
)

pixelTracksSequence = cms.Sequence(pixelTracksTask)


# Pixel triplets for trackingLowPU
pixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "pixelTracksHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(
        clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"
    )
)

trackingLowPU.toModify(pixelTracks,
    SeedingHitSets = "pixelTracksHitTriplets"
)

_pixelTracksTask_lowPU = pixelTracksTask.copy()
_pixelTracksTask_lowPU.replace(pixelTracksHitQuadruplets, pixelTracksHitTriplets)
trackingLowPU.toReplaceWith(pixelTracksTask, _pixelTracksTask_lowPU)


# "Patatrack" pixel ntuplets, fishbone cleaning, Broken Line fit, and density-based vertex reconstruction
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit

from RecoPixelVertexing.PixelTriplets.pixelTracksCUDA_cfi import pixelTracksCUDA as _pixelTracksCUDA

# SwitchProducer providing the pixel tracks in SoA format on the CPU
pixelTracksSoA = SwitchProducerCUDA(
    # build pixel ntuplets and pixel tracks in SoA format on the CPU
    cpu = _pixelTracksCUDA.clone(
        pixelRecHitSrc = "siPixelRecHitsPreSplitting",
        idealConditions = False,
        onGPU = False
    )
)
# use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
run3_common.toModify(pixelTracksSoA.cpu,
    idealConditions = True
)

# convert the pixel tracks from SoA to legacy format
from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import pixelTrackProducerFromSoA as _pixelTrackProducerFromSoA
(pixelNtupletFit & ~phase2_tracker).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoA.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

(pixelNtupletFit & ~phase2_tracker).toReplaceWith(pixelTracksTask, cms.Task(
    # build the pixel ntuplets and the pixel tracks in SoA format on the GPU
    pixelTracksSoA,
    # convert the pixel tracks from SoA to legacy format
    pixelTracks
))


# "Patatrack" sequence running on GPU (or CPU if not available)
from Configuration.ProcessModifiers.gpu_cff import gpu

(pixelNtupletFit & gpu).toModify(pixelTracksSoA.cpu,
    pixelRecHitSrc = "siPixelRecHitsPreSplittingSoA")

# build the pixel ntuplets and pixel tracks in SoA format on the GPU
pixelTracksCUDA = _pixelTracksCUDA.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingCUDA",
    idealConditions = False,
    onGPU = True
)
# use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
run3_common.toModify(pixelTracksCUDA,
    idealConditions = True
)

# SwitchProducer providing the pixel tracks in SoA format on the CPU
from RecoPixelVertexing.PixelTrackFitting.pixelTracksSoA_cfi import pixelTracksSoA as _pixelTracksSoA
gpu.toModify(pixelTracksSoA,
    # transfer the pixel tracks in SoA format to the host
    cuda = _pixelTracksSoA.clone()
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

(pixelNtupletFit & gpu & ~phase2_tracker).toReplaceWith(pixelTracksTask, cms.Task(
    # build the pixel ntuplets and pixel tracks in SoA format on the GPU
    pixelTracksCUDA,
    # transfer the pixel tracks in SoA format to the CPU, and convert them to legacy format
    pixelTracksTask.copy()
))

## GPU vs CPU validation
# force CPU vertexing to use hit SoA from CPU chain and not the converted one from GPU chain
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
(pixelNtupletFit & gpu & gpuValidationPixel).toModify(pixelTracksSoA.cpu,
    pixelRecHitSrc = "siPixelRecHitsPreSplittingSoA@cpu"
    )
