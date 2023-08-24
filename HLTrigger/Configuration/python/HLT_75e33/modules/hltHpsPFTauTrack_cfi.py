import FWCore.ParameterSet.Config as cms

hltHpsPFTauTrack = cms.EDFilter( "HLT1PFTau",
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHpsPFTauProducer" ),
    triggerType = cms.int32( 84 ),
    MinE = cms.double( -1.0 ),
    MinPt = cms.double( 0.0 ),
    MinMass = cms.double( -1.0 ),
    MaxMass = cms.double( -1.0 ),
    MinEta = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinN = cms.int32( 1 )
)
