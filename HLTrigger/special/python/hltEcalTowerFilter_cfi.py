import FWCore.ParameterSet.Config as cms

hltEcalTowerFilter = cms.EDFilter( "HLTEcalTowerFilter",
   inputTag = cms.InputTag( "hltTowerMakerForEcal" ),
   saveTag  = cms.untracked.bool( False ),
   MinE   = cms.double( 10.0 ),
   MaxEta = cms.double( 3.0 ),
   MinN   = cms.uint32( 1 )
)
