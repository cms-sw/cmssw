import FWCore.ParameterSet.Config as cms

#IsolatedPixelTrackCandidateProducer default configuration
isolPixelTrackProd    = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
    L1eTauJetsSource           = cms.InputTag('hltCaloStage2Digis','Tau'),
    tauAssociationCone         = cms.double( 0.0 ),
    tauUnbiasCone              = cms.double( 1.2 ),
    PixelTracksSources         = cms.VInputTag( "hltPixelTracks" ),
    ExtrapolationConeSize      = cms.double(1.0),
    PixelIsolationConeSizeAtEC = cms.double(40),
    L1GTSeedLabel              = cms.InputTag( "hltL1sIsoTrack" ),
    MaxVtxDXYSeed              = cms.double( 101.0 ),
    MaxVtxDXYIsol              = cms.double( 101.0 ),
    VertexLabel                = cms.InputTag( "hltTrimmedPixelVertices" ),
    MagFieldRecordName         = cms.string("VolumeBasedMagneticField"),
    minPTrack                  = cms.double( 5.0 ),
    maxPTrackForIsolation      = cms.double( 3.0 ),
    EBEtaBoundary              = cms.double(1.479)
)


