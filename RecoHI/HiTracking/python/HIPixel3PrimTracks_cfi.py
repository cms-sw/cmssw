import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

hiPixel3PrimTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet primary tracks with vertex constraint'),

    # Region
    RegionFactoryPSet = cms.PSet(
	  ComponentName = cms.string("GlobalTrackingRegionWithVerticesProducer"),
	  RegionPSet = cms.PSet(
                HiTrackingRegionWithVertexBlock
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
    Filter = cms.InputTag("hiFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)

# increase threshold for triplets in generation step (default: 10000)
hiPixel3PrimTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 1000000

hiPixel3PrimTracksSequence = cms.Sequence(
    pixelFitterByHelixProjections +
    hiFilter +
    hiPixel3PrimTracks
)
