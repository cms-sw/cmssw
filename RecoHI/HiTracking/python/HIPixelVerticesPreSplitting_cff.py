import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HIPixelVertices_cff import *

hiPixelClusterVertexPreSplitting = hiPixelClusterVertex.clone( pixelRecHits=cms.string("siPixelRecHitsPreSplitting") )
hiPixel3ProtoTracksPreSplitting = hiPixel3ProtoTracks.clone()
hiPixel3ProtoTracksPreSplitting.RegionFactoryPSet.RegionPSet.siPixelRecHits = cms.InputTag( "siPixelRecHitsPreSplitting" )
hiPixel3ProtoTracksPreSplitting.RegionFactoryPSet.RegionPSet.VertexCollection = cms.InputTag( "hiPixelClusterVertexPreSplitting" )
hiPixel3ProtoTracksPreSplitting.FilterPSet.siPixelRecHits = cms.InputTag( "siPixelRecHitsPreSplitting" )
hiPixel3ProtoTracksPreSplitting.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag( "PixelLayerTripletsPreSplitting" )

hiPixelMedianVertexPreSplitting = hiPixelMedianVertex.clone( TrackCollection = cms.InputTag('hiPixel3ProtoTracksPreSplitting') )
hiSelectedProtoTracksPreSplitting = hiSelectedProtoTracks.clone(
  src = cms.InputTag("hiPixel3ProtoTracksPreSplitting"),
  VertexCollection = cms.InputTag("hiPixelMedianVertexPreSplitting")
)
hiPixelAdaptiveVertexPreSplitting = hiPixelAdaptiveVertex.clone(
  TrackLabel = cms.InputTag("hiSelectedProtoTracksPreSplitting")
)
hiBestAdaptiveVertexPreSplitting = hiBestAdaptiveVertex.clone( src = cms.InputTag("hiPixelAdaptiveVertexPreSplitting") ) 
hiSelectedVertexPreSplitting = hiSelectedVertex.clone( 
  adaptiveVertexCollection = cms.InputTag("hiBestAdaptiveVertexPreSplitting"),
  medianVertexCollection = cms.InputTag("hiPixelMedianVertexPreSplitting")
)
bestHiVertexPreSplitting = cms.Sequence( hiBestAdaptiveVertexPreSplitting * hiSelectedVertexPreSplitting )

PixelLayerTripletsPreSplitting = PixelLayerTriplets.clone()
PixelLayerTripletsPreSplitting.FPix.HitProducer = 'siPixelRecHitsPreSplitting'
PixelLayerTripletsPreSplitting.BPix.HitProducer = 'siPixelRecHitsPreSplitting'

hiPixelVerticesPreSplitting = cms.Sequence(hiPixelClusterVertexPreSplitting
                                * PixelLayerTripletsPreSplitting
                                * hiPixel3ProtoTracksPreSplitting
                                * hiPixelMedianVertexPreSplitting
                                * hiSelectedProtoTracksPreSplitting
                                * hiPixelAdaptiveVertexPreSplitting
                                * bestHiVertexPreSplitting )
