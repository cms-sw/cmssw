import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoTracker.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi import PixelLayerQuadruplets as _PixelLayerQuadruplets

#from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

hiPixelLayerQuadruplets = _PixelLayerQuadruplets.clone()

# Hit ntuplets
hiPixel3PrimTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck    = "",
    seedingLayers   = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiPixel3PrimTracksHitDoublets,
    seedingLayers = "hiPixelLayerQuadruplets"
)


hiPixel3PrimTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets   = "hiPixel3PrimTracksHitDoublets",
    maxElement = 1000000, # increase threshold for triplets in generation step (default: 100000)
    produceSeedingHitSets = True,
    produceIntermediateHitTriplets = True,
)

from RecoTracker.PixelSeeding.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
hiPixel3PrimTracksHitDoubletsCA = hiPixel3PrimTracksHitDoublets.clone(
    layerPairs = [0,1,2]
)
hiPixel3PrimTracksHitQuadrupletsCA = _caHitQuadrupletEDProducer.clone(
    doublets = "hiPixel3PrimTracksHitDoubletsCA",
    extraHitRPhitolerance = hiPixel3PrimTracksHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet    = hiPixel3PrimTracksHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 200, value2 = 50,
    ),
    useBendingCorrection = True,
    fitFastCircle        = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0012,
    CAPhiCut   = 0.2,
) 

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod 

# Pixel tracks
hiPixel3PrimTracks = _mod.pixelTracks.clone(
    passLabel  = 'Pixel triplet primary tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiPixel3PrimTracksHitTriplets",
    # Fitter
    Fitter = "pixelFitterByHelixProjections",
    # Filter
    Filter = "hiFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)
trackingPhase1.toModify(hiPixel3PrimTracks,
    SeedingHitSets = "hiPixel3PrimTracksHitQuadrupletsCA",
)

hiPixel3PrimTracksTask = cms.Task(
    hiTrackingRegionWithVertex ,
    hiPixel3PrimTracksHitDoublets ,
    hiPixel3PrimTracksHitTriplets ,
    pixelFitterByHelixProjections ,
    hiFilter ,
    hiPixel3PrimTracks
)
hiPixel3PrimTracksSequence = cms.Sequence(hiPixel3PrimTracksTask)
#phase 1 changes
hiPixel3PrimTracksTask_Phase1 = hiPixel3PrimTracksTask.copy()
hiPixel3PrimTracksTask_Phase1.replace(hiPixel3PrimTracksHitDoublets, cms.Task(hiPixelLayerQuadruplets,hiPixel3PrimTracksHitDoubletsCA) )
hiPixel3PrimTracksTask_Phase1.replace(hiPixel3PrimTracksHitTriplets,hiPixel3PrimTracksHitQuadrupletsCA)
trackingPhase1.toReplaceWith(hiPixel3PrimTracksTask,hiPixel3PrimTracksTask_Phase1)
