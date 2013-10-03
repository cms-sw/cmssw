import FWCore.ParameterSet.Config as cms

# global tracking region for primary pixel tracks
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


# global tracking region for low-pt pixel tracks
HiLowPtTrackingRegionWithVertexBlock = cms.PSet(
    ptMin         = cms.double(0.2),
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

# limited tracking region for pixel proto-tracks passed to vertexing
HiTrackingRegionForPrimaryVertexBlock = cms.PSet( 
    ptMin = cms.double( 0.7 ),
    doVariablePtMin = cms.bool ( True ),
    originRadius = cms.double( 0.1 ),	
    nSigmaZ = cms.double(3.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),	
    precise = cms.bool( True ),
    siPixelRecHits = cms.InputTag( "siPixelRecHits" ),
    directionXCoord = cms.double( 1.0 ),
    directionYCoord = cms.double( 1.0 ),
    directionZCoord = cms.double( 0.0 )
    )

# limited tracking region for pixel proto-tracks (using cluster vtx input)
HiTrackingRegionFromClusterVtxBlock = cms.PSet( 
    ptMin = cms.double( 0.7 ),
    doVariablePtMin = cms.bool ( True ),
    originRadius = cms.double( 0.2 ),	
    nSigmaZ = cms.double(3.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),	
    precise = cms.bool( True ),
    siPixelRecHits = cms.InputTag( "siPixelRecHits" ),
    directionXCoord = cms.double( 1.0 ),
    directionYCoord = cms.double( 1.0 ),
    directionZCoord = cms.double( 0.0 ),
    useFoundVertices = cms.bool(True),
    VertexCollection = cms.InputTag("hiPixelClusterVertex"),		
    useFixedError = cms.bool(True),
    fixedError    = cms.double(3.0),
    sigmaZVertex  = cms.double(3.0)
    )

# limited jet-seeded tracking region
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import tauRegionalPixelSeedGenerator
HiTrackingRegionFactoryFromJetsBlock = tauRegionalPixelSeedGenerator.RegionFactoryPSet
HiTrackingRegionFactoryFromJetsBlock.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
HiTrackingRegionFactoryFromJetsBlock.RegionPSet.vertexSrc = cms.InputTag("hiSelectedVertex")

# limited stand-alone muon-seeded tracking region
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import MuonTrackingRegionCommon
HiTrackingRegionFactoryFromSTAMuonsBlock = cms.PSet(
    MuonServiceProxy,
    MuonTrackingRegionCommon,
    ComponentName = cms.string('HIMuonTrackingRegionProducer'),
    MuonSrc = cms.InputTag("standAloneMuons","UpdatedAtVtx")
    )
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex = cms.bool(True)
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Rescale_Dz = cms.double(5.0)
