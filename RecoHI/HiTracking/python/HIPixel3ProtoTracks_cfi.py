import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelSeeding.PixelTripletHLTGenerator_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoTracker.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

# Hit ntuplets
hiPixel3ProtoTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionFromClusterVtx",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)

hiPixel3ProtoTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiPixel3ProtoTracksHitDoublets",
    maxElement = 100000,
    produceSeedingHitSets = True,
)

import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod

# Pixel tracks
hiPixel3ProtoTracks = _mod.pixelTracks.clone( 
    passLabel  = 'Pixel triplet tracks for vertexing',
    # Ordered Hits
    SeedingHitSets = "hiPixel3ProtoTracksHitTriplets",
    # Fitter
    Fitter = "pixelFitterByHelixProjections",
    # Filter
    Filter = "hiProtoTrackFilter",
    # Cleaner
    Cleaner = "pixelTrackCleanerBySharedHits"
)

hiPixel3ProtoTracksTask = cms.Task(
    hiTrackingRegionFromClusterVtx ,
    hiPixel3ProtoTracksHitDoublets ,
    hiPixel3ProtoTracksHitTriplets ,
    pixelFitterByHelixProjections ,
    hiProtoTrackFilter ,
    hiPixel3ProtoTracks
)
hiPixel3ProtoTracksSequence = cms.Sequence(hiPixel3ProtoTracksTask)
