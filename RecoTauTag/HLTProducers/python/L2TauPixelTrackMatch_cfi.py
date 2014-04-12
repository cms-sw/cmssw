import FWCore.ParameterSet.Config as cms

L2TauPixelTrackMatch = cms.EDProducer( "L2TauPixelTrackMatch",
    JetSrc = cms.InputTag( "hltCaloJetPairDzMatchFilter" ),
    JetMinPt = cms.double( 25. ),
    JetMaxEta = cms.double( 2.4 ),
    TrackSrc = cms.InputTag( "hltPixelTracks" ),
    TrackMinPt = cms.double( 5. ),
    deltaR = cms.double( 0.2 ),
    BeamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" )
)

