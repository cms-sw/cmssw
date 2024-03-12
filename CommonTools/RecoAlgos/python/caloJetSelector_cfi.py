import FWCore.ParameterSet.Config as cms

caloJetSelector= cms.EDFilter( "CaloJetSelector",
    src = cms.InputTag( "JetCollection" ),
    filter = cms.bool( False ),
    cut = cms.string( "abs(eta)<3" )
)

# foo bar baz
# UzHMY0anNA7Ec
