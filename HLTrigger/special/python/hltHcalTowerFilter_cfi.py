import FWCore.ParameterSet.Config as cms

hltHcalTowerFilter = cms.EDFilter( "HLTHcalTowerFilter",
   inputTag = cms.InputTag( "hltTowerMakerForHcal" ),
   saveTag  = cms.untracked.bool( False ),
   MinE   = cms.double( 5.0 ),
#   MaxEta = cms.double( 3.0 ),
   MaxN   = cms.int32( 20 )
)
