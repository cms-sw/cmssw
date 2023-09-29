import FWCore.ParameterSet.Config as cms

hltHpsDoublePFTau40TrackPt1MediumChargedIsolation = cms.EDFilter( "HLT1PFTau",
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHpsSelectedPFTausTrackPt1MediumChargedIsolation" ),
    triggerType = cms.int32( 84 ),
    MinE = cms.double( -1.0 ),
    MinPt = cms.double( 40.0 ),
    MinMass = cms.double( -1.0 ),
    MaxMass = cms.double( -1.0 ),
    MinEta = cms.double( -1.0 ),
    MaxEta = cms.double( 2.1 ),
    MinN = cms.int32( 2 )
)
