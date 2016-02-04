import FWCore.ParameterSet.Config as cms

hltEcalTowerFilter = cms.EDFilter( "HLTEcalTowerFilter",
   inputTag = cms.InputTag( "hltTowerMakerForEcal" ),
   saveTags = cms.bool( False ),
   MinE   = cms.double( 10.0 ),
   MaxEta = cms.double( 3.0 ),
   MinN   = cms.int32( 1 )
)
