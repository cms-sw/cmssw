import FWCore.ParameterSet.Config as cms

hltBSoftMuonL25Jets = cms.EDFilter( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltSelector4Jets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 20.0 )
)

