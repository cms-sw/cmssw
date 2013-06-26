import FWCore.ParameterSet.Config as cms

hltCaloTowerFilter = cms.EDFilter( "HLTCaloTowerFilter",
   inputTag = cms.InputTag( "hltTowerMakerForEcal" ),
   saveTags = cms.bool( False ),
   MinPt  = cms.double( 3.0 ),
   MaxEta = cms.double( 3.0 ),
   MinN   = cms.uint32( 1 )
)
