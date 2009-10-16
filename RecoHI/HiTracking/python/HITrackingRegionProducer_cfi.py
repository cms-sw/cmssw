import FWCore.ParameterSet.Config as cms

HiTrackingRegionWithVertexBlock = cms.PSet(
    ptMin         = cms.double(1.5),	  
    originRadius  = cms.double(0.2),
    nSigmaZ       = cms.double(3.0),		
    beamSpot      = cms.InputTag("offlineBeamSpot"),
    precise       = cms.bool(True),		
    useFoundVertices = cms.bool(True),
    VertexCollection = cms.InputTag("hiSelectedVertex"),		
    useFixedError = cms.bool(True),
    fixedError    = cms.double(0.2),
    sigmaZVertex  = cms.double(3.0)		
    )

HiTrackingRegionForPrimaryVertexBlock = cms.PSet( 
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
