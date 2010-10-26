import FWCore.ParameterSet.Config as cms

hltExclDijet = cms.EDFilter( "HLTExclDiJetFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minPtJet = cms.double( 30.0 ),
    minHFe = cms.double( 50.0 ),
    HF_OR = cms.bool( False )
)
