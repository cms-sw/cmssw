import FWCore.ParameterSet.Config as cms

hltMuTrackJpsiPixelTrackSelector = cms.EDProducer( "QuarkoniaTrackSelector",
    muonCandidates = cms.InputTag( "hltL3MuonCandidates" ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    checkCharge = cms.bool( False ),
    MinTrackPt = cms.double( 0.0 ),
    MinTrackP = cms.double( 3.0 ),
    MaxTrackEta = cms.double( 999.0 ),
    MinMasses = cms.vdouble( 1.6 ),
    MaxMasses = cms.vdouble( 4.6 )
)

