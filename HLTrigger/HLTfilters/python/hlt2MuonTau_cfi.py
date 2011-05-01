import FWCore.ParameterSet.Config as cms

hlt2MuonTau= cms.EDFilter( "HLT2MuonTau",
    inputTag1 = cms.InputTag( "hlt1muon50" ),
    inputTag2 = cms.InputTag( "hlt1tau100" ),
    saveTags = cms.bool( True ),
    MinDphi = cms.double( 0.0 ),
    MaxDphi = cms.double( 2.1 ),
    MinDeta = cms.double( 0.0 ),
    MaxDeta = cms.double( -1.0 ),
    MinMinv = cms.double( 0.0 ),
    MaxMinv = cms.double( -1.0 ),
    MinDelR = cms.double( 0.0 ),
    MaxDelR = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)

