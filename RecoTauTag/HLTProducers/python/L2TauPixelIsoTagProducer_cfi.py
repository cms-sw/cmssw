import FWCore.ParameterSet.Config as cms


L2TauPixelIsoTagProducer =  cms.EDProducer( "L2TauPixelIsoTagProducer",
    JetSrc = cms.InputTag( "hltL2DiTauCaloJets" ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    TrackSrc = cms.InputTag( "" ), # not used yet
    BeamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxNumberPV = cms.int32( 1 ), # not used yet
    TrackMinPt = cms.double( 1.6 ),
    TrackMaxDxy = cms.double( 0.2 ),
    TrackMaxNChi2 = cms.double( 100. ),
    TrackMinNHits = cms.int32( 3 ),
    TrackPVMaxDZ = cms.double( 0.1 ), # not used yet
    IsoConeMin = cms.double( 0.2 ),
    IsoConeMax = cms.double( 0.4 )
)

