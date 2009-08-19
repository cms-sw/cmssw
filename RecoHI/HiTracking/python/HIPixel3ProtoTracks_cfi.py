import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *

hiPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet tracks for vertexing'),
	
	# Region
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
		ptMin = cms.double( 0.7 ),
        originRadius = cms.double( 0.1 ),	
		nSigmaZ = cms.double(3.0),
		beamSpot = cms.InputTag("offlineBeamSpot"),			
        precise = cms.bool( True ),
        siPixelRecHits = cms.string( "siPixelRecHits" ),
        directionXCoord = cms.double( 1.0 ),
        directionYCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 )
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
      ptMin = cms.double( 0.7 ),
      tipMax = cms.double( 1.0 ),
	  chi2 = cms.double( 1000.0 )
    ),
	
	# Cleaner
    CleanerPSet = cms.PSet(  
	  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) 
	)
)