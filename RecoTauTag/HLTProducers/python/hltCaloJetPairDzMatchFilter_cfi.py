import FWCore.ParameterSet.Config as cms

hltCaloJetPairDzMatchFilter = cms.EDFilter( "HLTCaloJetPairDzMatchFilter",
    saveTags = cms.bool( True ),
    JetSrc = cms.InputTag( "hltMatchL2Tau30ToPixelTrk5" ),
    JetMinPt = cms.double( 25. ),
    JetMaxEta = cms.double( 2.4 ),
    JetMinDR = cms.double( 0.2 ),
    JetMaxDZ = cms.double( 0.2 )
)

