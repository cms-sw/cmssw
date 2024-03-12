import FWCore.ParameterSet.Config as cms

etMinCaloJetSelector = cms.EDFilter( "EtMinPFJetSelector",
    src = cms.InputTag( "JetCollection" ),
    filter = cms.bool( False ),
    etMin = cms.double( 20.0 )
)

# foo bar baz
# h1Nn3X9fausbe
# 2kI5d7IEl8nME
