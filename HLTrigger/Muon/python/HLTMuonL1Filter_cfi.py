import FWCore.ParameterSet.Config as cms

HLTMuonL1Filter = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( '' ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SelectQualities = cms.vint32( ),
    CSCTFtag = cms.InputTag("csctfDigis"),
    ExcludeSingleSegmentCSC = cms.bool(False),
    SaveTag = cms.untracked.bool( False )
)

