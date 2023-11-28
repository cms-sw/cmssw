import FWCore.ParameterSet.Config as cms

hltHpsDoublePFTau35MediumDitauWPDeepTau = cms.EDFilter( "HLT1PFTau",
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHpsSelectedPFTausMediumDitauWPDeepTau" ),
    triggerType = cms.int32( 84 ),
    MinE = cms.double( -1.0 ),
    MinPt = cms.double( 35.0 ),
    MinMass = cms.double( -1.0 ),
    MaxMass = cms.double( -1.0 ),
    MinEta = cms.double( -1.0 ),
    MaxEta = cms.double( 2.1 ),
    MinN = cms.int32( 2 )
)
