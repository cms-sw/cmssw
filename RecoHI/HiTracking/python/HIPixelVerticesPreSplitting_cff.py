import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HIPixelVertices_cff import *

hiPixelClusterVertexPreSplitting = hiPixelClusterVertex.clone(
    pixelRecHits = "siPixelRecHitsPreSplitting" 
)

hiProtoTrackFilterPreSplitting = hiProtoTrackFilter.clone(
    siPixelRecHits = "siPixelRecHitsPreSplitting"
)

hiPixel3ProtoTracksTrackingRegionsPreSplitting = hiTrackingRegionFromClusterVtx.clone(
    RegionPSet = dict(
        siPixelRecHits   = "siPixelRecHitsPreSplitting",
        VertexCollection = "hiPixelClusterVertexPreSplitting"
    )
)

hiPixel3PRotoTracksHitDoubletsPreSplitting = hiPixel3ProtoTracksHitDoublets.clone(
    seedingLayers = "PixelLayerTripletsPreSplitting",
    trackingRegions = "hiPixel3ProtoTracksTrackingRegionsPreSplitting",
)

hiPixel3ProtoTracksHitTripletsPreSplitting = hiPixel3ProtoTracksHitTriplets.clone(
    doublets = "hiPixel3PRotoTracksHitDoubletsPreSplitting"
)

hiPixel3ProtoTracksPreSplitting = hiPixel3ProtoTracks.clone(
    SeedingHitSets = "hiPixel3ProtoTracksHitTripletsPreSplitting",
    Filter         = "hiProtoTrackFilterPreSplitting",
)

hiPixelMedianVertexPreSplitting = hiPixelMedianVertex.clone(
    TrackCollection = 'hiPixel3ProtoTracksPreSplitting'
)

hiSelectedProtoTracksPreSplitting = hiSelectedProtoTracks.clone(
    src              = "hiPixel3ProtoTracksPreSplitting",
    VertexCollection = "hiPixelMedianVertexPreSplitting"
)

hiPixelAdaptiveVertexPreSplitting = hiPixelAdaptiveVertex.clone(
    TrackLabel = "hiSelectedProtoTracksPreSplitting"
)

hiBestAdaptiveVertexPreSplitting = hiBestAdaptiveVertex.clone(
    src = "hiPixelAdaptiveVertexPreSplitting"
)

hiSelectedVertexPreSplitting = hiSelectedPixelVertex.clone( 
    adaptiveVertexCollection = "hiBestAdaptiveVertexPreSplitting",
    medianVertexCollection   = "hiPixelMedianVertexPreSplitting"
)
bestHiVertexPreSplittingTask = cms.Task( hiBestAdaptiveVertexPreSplitting , hiSelectedVertexPreSplitting )

PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone(
    FPix = dict(HitProducer = 'siPixelRecHitsPreSplitting'),
    BPix = dict(HitProducer = 'siPixelRecHitsPreSplitting')
)
hiPixelVerticesPreSplittingTask = cms.Task(hiPixelClusterVertexPreSplitting
                                , PixelLayerTripletsPreSplitting
                                , hiPixel3ProtoTracksTrackingRegionsPreSplitting
                                , hiPixel3PRotoTracksHitDoubletsPreSplitting
                                , hiPixel3ProtoTracksHitTripletsPreSplitting
                                , hiProtoTrackFilterPreSplitting
                                , pixelFitterByHelixProjections
                                , hiPixel3ProtoTracksPreSplitting
                                , hiPixelMedianVertexPreSplitting
                                , hiSelectedProtoTracksPreSplitting
                                , hiPixelAdaptiveVertexPreSplitting
                                , bestHiVertexPreSplittingTask )
hiPixelVerticesPreSplitting = cms.Sequence(hiPixelVerticesPreSplittingTask)
