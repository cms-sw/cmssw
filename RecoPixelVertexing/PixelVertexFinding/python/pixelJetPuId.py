import FWCore.ParameterSet.Config as cms

jetNoPUPixelTag = cms.EDFilter( "PixelJetPuId",
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    jets = cms.InputTag( "hltCaloJetL1FastJetCorrected" ),
    tracks = cms.InputTag( "hltPixelTracksNoPU" ), ##regional pixel tracks around jets and with primary vertex constraint
    primaryVertex = cms.InputTag( "hltFastPVPixelVertices" ),

    MinGoodJetTrackPtRatio = cms.double( 0.045 ), 
    MinGoodJetTrackPt = cms.double( 1.8 ), 

    MaxTrackDistanceToJet = cms.double( 0.04 ), 
    MinTrackPt = cms.double( 0.6 ), 
    MaxTrackChi2 = cms.double( 20 ),

    UseForwardJetsAsNoPU = cms.bool( True ),
    MinEtaForwardJets = cms.double( 2.4 ),
    MinEtForwardJets = cms.double( 40 )
      
)

