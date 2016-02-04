import FWCore.ParameterSet.Config as cms

hltHcalTowerFilter = cms.EDFilter( "HLTHcalTowerFilter",
   inputTag  = cms.InputTag( "hltTowerMakerForHcal" ),
   saveTags = cms.bool( False ),
   MinE_HB   = cms.double( 1.5 ),
   MinE_HE   = cms.double( 2.5 ),
   MinE_HF   = cms.double( 9.0 ),
   MaxN_HB   = cms.int32( 2 ),
   MaxN_HE   = cms.int32( 2 ),
   MaxN_HF   = cms.int32( 8 )                                
)
