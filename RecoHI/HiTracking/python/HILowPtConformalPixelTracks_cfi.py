import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoTracker.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByConformalMappingAndLine_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

# Hit ntuplets
hiConformalPixelTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck    = "",
    seedingLayers   = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)

hiConformalPixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets   = "hiConformalPixelTracksHitDoublets",
    maxElement = 5000000, # increase threshold for triplets in generation step (default: 100000)
    produceSeedingHitSets = True,
)

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod
# Pixel tracks
hiConformalPixelTracks = _mod.pixelTracks.clone(
    #passLabel  = 'Pixel triplet low-pt tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiConformalPixelTracksHitTriplets",
    # Fitter
    Fitter = 'pixelFitterByConformalMappingAndLine',
    # Filter
    Filter = "hiConformalPixelFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)

###Pixel Tracking -  PhaseI geometry

#Tracking regions - use PV from pp tracking
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices
hiConformalPixelTracksPhase1TrackingRegions = globalTrackingRegionWithVertices.clone(
    RegionPSet = dict(
	precise = True,
	useMultipleScattering = False,
	useFakeVertices  = False,
	beamSpot         = "offlineBeamSpot",
	useFixedError    = True,
	nSigmaZ          = 3.0,
	sigmaZVertex     = 3.0,
	fixedError       = 0.2,
	VertexCollection = "offlinePrimaryVertices",
	ptMin            = 0.3,
	useFoundVertices = True,
	originRadius     = 0.2
    )
)

# SEEDING LAYERS
# Using 4 layers layerlist
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepSeedLayers
hiConformalPixelTracksPhase1SeedLayers = lowPtQuadStepSeedLayers.clone(
    BPix = cms.PSet(
	HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('WithTrackAngle'),
    ),
    FPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('WithTrackAngle'),
    )
)


# Hit ntuplets
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepHitDoublets
hiConformalPixelTracksPhase1HitDoubletsCA = lowPtQuadStepHitDoublets.clone(
    seedingLayers   = "hiConformalPixelTracksPhase1SeedLayers",
    trackingRegions = "hiConformalPixelTracksPhase1TrackingRegions"
)


from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepHitQuadruplets
hiConformalPixelTracksPhase1HitQuadrupletsCA = lowPtQuadStepHitQuadruplets.clone(
    doublets   = "hiConformalPixelTracksPhase1HitDoubletsCA",
    CAPhiCut   = 0.2,
    CAThetaCut = 0.0012,
    SeedComparitorPSet = dict(
       ComponentName = 'none'
    ),
    extraHitRPhitolerance = 0.032,
    maxChi2 = dict(
       enabled = True,
       pt1     = 0.7,
       pt2     = 2,
       value1  = 200,
       value2  = 50
    )
)

#Filter
hiConformalPixelTracksPhase1Filter = hiConformalPixelFilter.clone(
    VertexCollection = "offlinePrimaryVertices",
    chi2   = 30.0,
    lipMax = 999.0,
    nSigmaLipMaxTolerance = 3.0,
    nSigmaTipMaxTolerance = 3.0,
    ptMax  = 999999,
    ptMin  = 0.30,
    tipMax = 999.0
)

from RecoTracker.PixelTrackFitting.pixelNtupletsFitter_cfi import pixelNtupletsFitter

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(hiConformalPixelTracks,
    Cleaner = 'pixelTrackCleanerBySharedHits',
    Filter  = "hiConformalPixelTracksPhase1Filter",
    Fitter  = "pixelNtupletsFitter",
    SeedingHitSets = "hiConformalPixelTracksPhase1HitQuadrupletsCA",
)

hiConformalPixelTracksTask = cms.Task(
    hiTrackingRegionWithVertex ,
    hiConformalPixelTracksHitDoublets ,
    hiConformalPixelTracksHitTriplets ,
    pixelFitterByConformalMappingAndLine ,
    hiConformalPixelFilter ,
    hiConformalPixelTracks
)

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAHIonPhase1_cfi import pixelTrackSoAFromCUDAHIonPhase1 as _pixelTracksSoA
from RecoTracker.PixelSeeding.caHitNtupletCUDAHIonPhase1_cfi import caHitNtupletCUDAHIonPhase1 as _pixelTracksCUDA
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAHIonPhase1_cfi import pixelTrackProducerFromSoAHIonPhase1 as _pixelTrackProducerFromSoA

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

hiPixelTracksCUDA = _pixelTracksCUDA.clone(pixelRecHitSrc="siPixelRecHitsPreSplittingCUDA", idealConditions = False,
        ptmin = 0.25, hardCurvCut = 0.0756, doPtCut = False,
        onGPU = True,
        phiCuts = cms.vint32(19*[900]), #19 pairs
        trackQualityCuts = dict(
          chi2MaxPt = 10,
          chi2Coeff = [0.9,1.8],
          chi2Scale = 8,
          tripletMinPt = 0.5,
          tripletMaxTip = 0.3,
          tripletMaxZip = 12,
          quadrupletMinPt = 0.3,
          quadrupletMaxTip = 0.5,
          quadrupletMaxZip = 12
        ))

# SwitchProducer providing the pixel tracks in SoA format on the CPU
hiPixelTracksSoA = SwitchProducerCUDA(
    # build pixel ntuplets and pixel tracks in SoA format on the CPU
    cpu = _pixelTracksCUDA.clone(
        pixelRecHitSrc = "siPixelRecHitsPreSplittingCPU",
        idealConditions = False,
    	doPtCut = False,
    	ptmin = 0.25,
    	hardCurvCut = 0.0756,
        onGPU = False,
        phiCuts = cms.vint32(19*[900]), #19 pairs
        trackQualityCuts = dict(
          chi2MaxPt = 10,
          chi2Coeff = [0.9,1.8],
          chi2Scale = 8,
          tripletMinPt = 0.5,
          tripletMaxTip = 0.3,
          tripletMaxZip = 12,
          quadrupletMinPt = 0.3,
          quadrupletMaxTip = 0.5,
          quadrupletMaxZip = 12
        ))
)

gpu.toModify(hiPixelTracksSoA,
    # transfer the pixel tracks in SoA format to the host
    cuda = _pixelTracksSoA.clone(src="hiPixelTracksCUDA")
)

pixelNtupletFit.toReplaceWith(hiConformalPixelTracks,_pixelTrackProducerFromSoA.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    trackSrc = "hiPixelTracksSoA",
    minQuality = "highPurity"
))


hiConformalPixelTracksTaskPhase1 = cms.Task(
    hiConformalPixelTracksPhase1TrackingRegions ,
    hiConformalPixelTracksPhase1SeedLayers ,
    hiConformalPixelTracksPhase1HitDoubletsCA ,
    hiConformalPixelTracksPhase1HitQuadrupletsCA ,
    pixelNtupletsFitter ,
    hiConformalPixelTracksPhase1Filter ,
    hiConformalPixelTracks
)

pixelNtupletFit.toReplaceWith(hiConformalPixelTracksTaskPhase1, cms.Task(
    # build the pixel ntuplets and the pixel tracks in SoA format on the CPU
    hiPixelTracksSoA,
    # convert the pixel tracks from SoA to legacy format
    hiConformalPixelTracks
))

(gpu & pixelNtupletFit).toReplaceWith(hiConformalPixelTracksTaskPhase1, cms.Task(
    # build the pixel ntuplets and the pixel tracks in SoA format on the GPU
    hiPixelTracksCUDA,
    # just copying the task above
    hiConformalPixelTracksTaskPhase1.copy()
))

hiConformalPixelTracksSequencePhase1 = cms.Sequence(hiConformalPixelTracksTaskPhase1)
