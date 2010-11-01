import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cfi import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

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
      SeedingLayers = cms.string( "PixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
	  PixelTripletHLTGenerator
      )
    ),
	
    # Fitter
    FitterPSet = cms.PSet( 
      ComponentName = cms.string('PixelFitterByHelixProjections'),
      TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    ),
	
    # Filter
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      HiProtoTrackFilterBlock
    ),
	
    # Cleaner
    CleanerPSet = cms.PSet(  
      ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) 
    )
)
