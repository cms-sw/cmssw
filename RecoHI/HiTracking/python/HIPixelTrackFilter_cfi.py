import FWCore.ParameterSet.Config as cms

ClusterFilterBlock = cms.PSet(
    ComponentName = cms.string( "ClusterShapeTrackFilter" ),
    ptMin = cms.double( 1.5 )
    )

HiFilterBlock = cms.PSet(
    ComponentName = cms.string( "HIPixelTrackFilter" ),
    ptMin = cms.double( 1.5 ),
    chi2 = cms.double( 1000.0 ),
    useClusterShape = cms.bool( False ),
    VertexCollection = cms.InputTag("hiSelectedVertex"),
    nSigmaTipMaxTolerance = cms.double( 6.0 ),
    tipMax = cms.double( 0 ),
    nSigmaLipMaxTolerance = cms.double( 0 ),
    lipMax = cms.double( 0.3 )
    )

KinematicFilterBlock = cms.PSet(
    nSigmaTipMaxTolerance = cms.double( 0.0 ),
    ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
    nSigmaInvPtTolerance = cms.double( 0.0 ),
    ptMin = cms.double( 0.7 ),
    tipMax = cms.double( 1.0 ),
    chi2 = cms.double( 1000.0 )
    )

HiProtoTrackFilterBlock = cms.PSet( 
    ComponentName = cms.string( "HIProtoTrackFilter" ),
    ptMin = cms.double( 1.0 ),
    doVariablePtMin = cms.bool( True ),
    tipMax = cms.double( 1.0 ),
    chi2 = cms.double( 1000.0 ),
    beamSpot = cms.InputTag( "offlineBeamSpot" ),
    siPixelRecHits = cms.InputTag( "siPixelRecHits" )
    )

HiConformalPixelFilterBlock = cms.PSet(
    ComponentName = cms.string( "HIPixelTrackFilter" ),
    ptMin = cms.double( 0.2 ),
    chi2 = cms.double( 80.0 ),
    useClusterShape = cms.bool( False ),
    VertexCollection = cms.InputTag("hiSelectedVertex"),
    nSigmaTipMaxTolerance = cms.double( 999.0 ),
    tipMax = cms.double( 999.0 ),
    nSigmaLipMaxTolerance = cms.double( 14.0 ),
    lipMax = cms.double( 999.0 )
    )
