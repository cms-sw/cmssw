import FWCore.ParameterSet.Config as cms

pfJetSelector= cms.EDFilter( "PFJetSelector",
    src = cms.InputTag( "JetCollection" ),
    filter = cms.bool( False ),
    cut = cms.string( "abs(eta)<3" )
)

