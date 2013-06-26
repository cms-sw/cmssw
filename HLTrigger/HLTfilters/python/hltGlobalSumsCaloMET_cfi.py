import FWCore.ParameterSet.Config as cms

hltGlobalSumsCaloMET= cms.EDFilter( "HLTGlobalSumsCaloMET",
    inputTag = cms.InputTag( "hltEcalMet" ),
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 30.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)

