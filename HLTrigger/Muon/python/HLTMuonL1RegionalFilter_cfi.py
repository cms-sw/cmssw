import FWCore.ParameterSet.Config as cms

HLTMuonL1RegionalFilter = cms.EDFilter( "HLTMuonL1RegionalFilter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu20" ),
    MinN = cms.int32( 1 ),
    saveTags = cms.bool( True ),
    Cuts = cms.VPSet(
        cms.PSet(
            EtaRange = cms.vdouble( -2.5, -1.6 ),
            MinPt  = cms.double( 20 ),
            QualityBits = cms.vuint32( 6, 7 )
        ),
        cms.PSet(
            EtaRange = cms.vdouble( -1.6,  1.6 ),
            MinPt  = cms.double( 20 ),
            QualityBits = cms.vuint32( 7 )
        ),
        cms.PSet(
            EtaRange = cms.vdouble(  1.6,  2.5 ),
            MinPt  = cms.double( 20 ),
            QualityBits = cms.vuint32( 6, 7 )
        )
    )
)

