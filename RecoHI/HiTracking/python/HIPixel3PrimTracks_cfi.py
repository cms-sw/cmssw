import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *

hiPixel3PrimTracks = cms.EDFilter("PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet primary tracks with vertex constraint'),

    # Region
    RegionFactoryPSet = cms.PSet(
	  ComponentName = cms.string("GlobalTrackingRegionWithVerticesProducer"),
	  RegionPSet = cms.PSet(
		ptMin         = cms.double(1.5),	  
		originRadius  = cms.double(0.2),
		nSigmaZ       = cms.double(3.0),		
		beamSpot      = cms.InputTag("offlineBeamSpot"),
		precise       = cms.bool(True),		
		useFoundVertices = cms.bool(True),
		VertexCollection = cms.InputTag("hiPixelAdaptiveVertex"),		
		useFixedError = cms.bool(True),
		fixedError    = cms.double(0.2),
		sigmaZVertex  = cms.double(3.0),		
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
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      ptMin = cms.double( 1.5 ),
      tipMax = cms.double( 0.2 ),
	  chi2 = cms.double( 1000.0 )
    ),
	
	# Cleaner
    CleanerPSet = cms.PSet(  
	  #ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) 
	  ComponentName = cms.string( "none" )
	)
)