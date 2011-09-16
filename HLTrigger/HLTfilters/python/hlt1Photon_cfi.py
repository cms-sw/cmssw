import FWCore.ParameterSet.Config as cms

hlt1Photon= cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)

