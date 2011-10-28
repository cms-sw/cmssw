import FWCore.ParameterSet.Config as cms

hltFilterDisplacedmumu = cms.EDFilter("HLTDisplacedmumumuFilter",
    MuonTag = cms.InputTag( "hltL3MuonCandidates" ),
    DisplacedVertexTag = cms.InputTag( "hltDisplacedmumumuVtx" ),
    FastAccept = cms.bool( False ),
    MinLxySignificance = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinVtxProbability = cms.double( 0.0 ),
    MinCosinePointingAngle = cms.double( -2.0 ),
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" )
)
