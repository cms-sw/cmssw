import FWCore.ParameterSet.Config as cms

# global tracking region for primary pixel tracks
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
hiTrackingRegionWithVertex = _globalTrackingRegionWithVertices.clone(RegionPSet = dict(
    ptMin         = cms.double(1.5),	  
    originRadius  = cms.double(0.2),
    nSigmaZ       = cms.double(3.0),		
    beamSpot      = cms.InputTag("offlineBeamSpot"),
    precise       = cms.bool(True),		
    useMultipleScattering = cms.bool(False),
    useFakeVertices       = cms.bool(False),
    useFoundVertices = cms.bool(True),
    VertexCollection = cms.InputTag("hiSelectedVertex"),		
    useFixedError = cms.bool(True),
    fixedError    = cms.double(0.2),
    sigmaZVertex  = cms.double(3.0)		
))

# global tracking region for low-pt pixel tracks
HiLowPtTrackingRegionWithVertexBlock = cms.PSet(
    ptMin         = cms.double(0.25),
    originRadius  = cms.double(0.2),
    nSigmaZ       = cms.double(3.0),
    beamSpot      = cms.InputTag("offlineBeamSpot"),
    precise       = cms.bool(True),
    useMultipleScattering = cms.bool(False),
    useFakeVertices       = cms.bool(False),
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
    useMultipleScattering = cms.bool(False),
    useFakeVertices       = cms.bool(False),
    siPixelRecHits = cms.InputTag( "siPixelRecHits" ),
    directionXCoord = cms.double( 1.0 ),
    directionYCoord = cms.double( 1.0 ),
    directionZCoord = cms.double( 0.0 )
    )

# limited tracking region for pixel proto-tracks (using cluster vtx input)
from RecoHI.HiTracking.hiTrackingRegionFromClusterVtx_cfi import hiTrackingRegionFromClusterVtx


# limited jet-seeded tracking region
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import tauRegionalPixelSeedGenerator
HiTrackingRegionFactoryFromJetsBlock = tauRegionalPixelSeedGenerator.RegionFactoryPSet
HiTrackingRegionFactoryFromJetsBlock.RegionPSet.JetSrc = cms.InputTag("iterativeConePu5CaloJets")
HiTrackingRegionFactoryFromJetsBlock.RegionPSet.vertexSrc = cms.InputTag("hiSelectedVertex")

# limited stand-alone muon-seeded tracking region
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import MuonTrackingRegionCommon
from RecoHI.HiMuonAlgos.HiTrackingRegionEDProducer_cfi import HiTrackingRegionEDProducer as _HiTrackingRegionEDProducer
HiTrackingRegionFactoryFromSTAMuonsEDProducer = _HiTrackingRegionEDProducer.clone(
    MuonSrc = "standAloneMuons:UpdatedAtVtx",
    MuonTrackingRegionBuilder = MuonTrackingRegionCommon.MuonTrackingRegionBuilder.clone(
        vertexCollection = "hiSelectedVertex",
        UseVertex = True,
        Rescale_Dz = 5.0,
    ),
    ServiceParameters = MuonServiceProxy.ServiceParameters,
)
