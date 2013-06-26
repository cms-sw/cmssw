import FWCore.ParameterSet.Config as cms

hltGlobalSumsMET= cms.EDFilter( "HLTGlobalSumsMET",
    inputTag = cms.InputTag( "hltJet15UHt" ),
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 100.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)

