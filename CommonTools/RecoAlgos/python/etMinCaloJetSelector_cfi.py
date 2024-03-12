import FWCore.ParameterSet.Config as cms

etMinCaloJetSelector = cms.EDFilter( "EtMinCaloJetSelector",
    src = cms.InputTag( "JetCollection" ),
    filter = cms.bool( False ),
    etMin = cms.double( 20.0 )
)

# foo bar baz
# rafK6qD8hwy4u
