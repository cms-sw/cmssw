import FWCore.ParameterSet.Config as cms

hltDisplacedMuMukFilter = cms.EDFilter( "HLTDisplacedmumuFilter",
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinPtPair = cms.double( 0.0 ),
    MinInvMass = cms.double( 0.2 ),
    MaxInvMass = cms.double( 3.0 ),
    ChargeOpt = cms.int32( 0 ),
    FastAccept = cms.bool( False ),
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    saveTags = cms.bool( True )
)
