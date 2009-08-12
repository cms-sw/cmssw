import FWCore.ParameterSet.Config as cms

#IsolatedPixelTrackCandidateProducer default configuration
isolPixelTrackProd = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
    L1eTauJetsSource = cms.InputTag( 'hltL1extraParticles','Tau' ),
    tauAssociationCone = cms.double( 0.0 ),
    tauUnbiasCone = cms.double( 1.2 ),
    PixelTracksSource = cms.InputTag( "hltPixelTracks" ),
    PixelIsolationConeSizeHB = cms.double( 0.3 ),
    PixelIsolationConeSizeHE = cms.double( 0.5 ),
    L1GTSeedLabel = cms.InputTag( "hltL1sIsoTrack" ),
    MaxVtxDXYSeed = cms.double( 0.05 ),
    MaxVtxDXYIsol = cms.double( 10.0 ),
    VertexLabel = cms.InputTag( "hltPixelVertices" ),
    minPtTrack = cms.double( 2.0 ),
    maxPtTrackForIsolation = cms.double( 5.0 )
)


