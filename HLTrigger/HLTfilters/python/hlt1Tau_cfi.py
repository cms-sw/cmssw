import FWCore.ParameterSet.Config as cms

hlt1Tau= cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltL25TauPixelTracksIsolationSelector" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)

