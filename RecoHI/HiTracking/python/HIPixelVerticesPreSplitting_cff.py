import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HIPixelVertices_cff import *

hiPixelClusterVertexPreSplitting = hiPixelClusterVertex.clone( pixelRecHits=cms.string("siPixelRecHitsPreSplitting") )

hiProtoTrackFilterPreSplitting = hiProtoTrackFilter.clone(
    siPixelRecHits = "siPixelRecHitsPreSplitting"
)

hiPixel3ProtoTracksTrackingRegionsPreSplitting = hiTrackingRegionFromClusterVtx.clone(RegionPSet=dict(
    siPixelRecHits = "siPixelRecHitsPreSplitting",
    VertexCollection = "hiPixelClusterVertexPreSplitting"
))
hiPixel3PRotoTracksHitDoubletsPreSplitting = hiPixel3ProtoTracksHitDoublets.clone(
    seedingLayers = "PixelLayerTripletsPreSplitting",
    trackingRegions = "hiPixel3ProtoTracksTrackingRegionsPreSplitting",
)
hiPixel3ProtoTracksHitTripletsPreSplitting = hiPixel3ProtoTracksHitTriplets.clone(
    doublets = "hiPixel3PRotoTracksHitDoubletsPreSplitting"
)

hiPixel3ProtoTracksPreSplitting = hiPixel3ProtoTracks.clone(
    SeedingHitSets = "hiPixel3ProtoTracksHitTripletsPreSplitting",
    Filter = "hiProtoTrackFilterPreSplitting",
)

hiPixelMedianVertexPreSplitting = hiPixelMedianVertex.clone( TrackCollection = cms.InputTag('hiPixel3ProtoTracksPreSplitting') )
hiSelectedProtoTracksPreSplitting = hiSelectedProtoTracks.clone(
  src = cms.InputTag("hiPixel3ProtoTracksPreSplitting"),
  VertexCollection = cms.InputTag("hiPixelMedianVertexPreSplitting")
)
hiPixelAdaptiveVertexPreSplitting = hiPixelAdaptiveVertex.clone(
  TrackLabel = cms.InputTag("hiSelectedProtoTracksPreSplitting")
)
hiBestAdaptiveVertexPreSplitting = hiBestAdaptiveVertex.clone( src = cms.InputTag("hiPixelAdaptiveVertexPreSplitting") ) 
hiSelectedVertexPreSplitting = hiSelectedPixelVertex.clone( 
  adaptiveVertexCollection = cms.InputTag("hiBestAdaptiveVertexPreSplitting"),
  medianVertexCollection = cms.InputTag("hiPixelMedianVertexPreSplitting")
)
bestHiVertexPreSplittingTask = cms.Task( hiBestAdaptiveVertexPreSplitting , hiSelectedVertexPreSplitting )

PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone()
PixelLayerTripletsPreSplitting.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
PixelLayerTripletsPreSplitting.BPix.HitProducer = 'siPixelRecHitsPreSplitting'

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
