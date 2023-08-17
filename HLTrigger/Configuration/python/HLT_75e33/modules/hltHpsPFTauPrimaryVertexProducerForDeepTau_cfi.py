import FWCore.ParameterSet.Config as cms

hltHpsPFTauPrimaryVertexProducerForDeepTau = cms.EDProducer( "PFTauPrimaryVertexProducer",
    qualityCuts = cms.PSet( 
      signalQualityCuts = cms.PSet( 
        minTrackPt = cms.double( 0.5 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.1 ),
        maxDeltaZ = cms.double( 0.4 ),
        maxDeltaZToLeadTrack = cms.double( -1.0 ),
        minTrackVertexWeight = cms.double( -1.0 ),
        minTrackPixelHits = cms.uint32( 0 ),
        minTrackHits = cms.uint32( 3 ),
        minGammaEt = cms.double( 1.0 ),
        minNeutralHadronEt = cms.double( 30.0 )
      ),
      isolationQualityCuts = cms.PSet( 
        minTrackPt = cms.double( 1.0 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.03 ),
        maxDeltaZ = cms.double( 0.2 ),
        maxDeltaZToLeadTrack = cms.double( -1.0 ),
        minTrackVertexWeight = cms.double( -1.0 ),
        minTrackPixelHits = cms.uint32( 0 ),
        minTrackHits = cms.uint32( 8 ),
        minGammaEt = cms.double( 1.5 )
      ),
      vxAssocQualityCuts = cms.PSet( 
        minTrackPt = cms.double( 0.5 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.1 ),
        minTrackVertexWeight = cms.double( -1.0 ),
        minTrackPixelHits = cms.uint32( 0 ),
        minTrackHits = cms.uint32( 3 ),
        minGammaEt = cms.double( 1.0 )
      ),
      primaryVertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),

      pvFindingAlgo = cms.string( "closestInDeltaZ" ),
      vertexTrackFiltering = cms.bool( False ),
      recoverLeadingTrk = cms.bool( False ),
      leadingTrkOrPFCandOption = cms.string( "leadPFCand" )
    ),
    cut = cms.string( "pt > 18.0 & abs(eta)<2.4" ),
    Algorithm = cms.int32( 0 ),
    RemoveElectronTracks = cms.bool( False ),
    RemoveMuonTracks = cms.bool( False ),
    useBeamSpot = cms.bool( True ),
    useSelectedTaus = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    ElectronTag = cms.InputTag( "hltEgammaCandidates" ),
    PFTauTag = cms.InputTag( "hltHpsPFTauProducer" ),

    MuonTag = cms.InputTag( "hltMuons" ),
    PVTag = cms.InputTag( "hltPhase2PixelVertices" ),

    discriminators = cms.VPSet( 
        cms.PSet(  discriminator = cms.InputTag( "hltHpsPFTauDiscriminationByDecayModeFindingNewDMs" ),
                   selectionCut = cms.double( 0.5 )
      )
    )
)
