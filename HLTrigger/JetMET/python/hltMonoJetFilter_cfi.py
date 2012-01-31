import FWCore.ParameterSet.Config as cms

hltMonoJetFilter = cms.EDFilter("HLTMonoJetFilter",
    inputJetTag = cms.InputTag( "hltCaloJetCorrectedRegionalHF" ),
    saveTags    = cms.bool( False ),
    max_PtSecondJet = cms.double( 9999. ),
    max_DeltaPhi = cms.double( 3.5 )
)

