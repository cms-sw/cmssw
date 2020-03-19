import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import *
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

# Pixel tracks
hiPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet tracks for vertexing'),
	
    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiPixel3ProtoTracksHitTriplets"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiProtoTrackFilter"),
	
    # Cleaner
    Cleaner = cms.string("pixelTrackCleanerBySharedHits")
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
