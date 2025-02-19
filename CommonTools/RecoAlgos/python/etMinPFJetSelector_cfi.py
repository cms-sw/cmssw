import FWCore.ParameterSet.Config as cms

etMinCaloJetSelector = cms.EDFilter( "EtMinPFJetSelector",
    src = cms.InputTag( "JetCollection" ),
    filter = cms.bool( False ),
    etMin = cms.double( 20.0 )
)

