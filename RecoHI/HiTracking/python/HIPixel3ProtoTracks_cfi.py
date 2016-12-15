import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

hiPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet tracks for vertexing'),
	
    # Region
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
          #HiTrackingRegionForPrimaryVertexBlock
          HiTrackingRegionFromClusterVtxBlock
      )
    ),
	
    # Ordered Hits
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.InputTag( "PixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
	  PixelTripletHLTGenerator
      )
    ),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiProtoTrackFilter"),
	
    # Cleaner
    Cleaner = cms.string("pixelTrackCleanerBySharedHits")
)

hiPixel3ProtoTracksSequence = cms.Sequence(
    pixelFitterByHelixProjections +
    hiProtoTrackFilter +
    hiPixel3ProtoTracks
)
