import FWCore.ParameterSet.Config as cms

hltAcoFilter= cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "IterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "MET" ),
    saveTags = cms.bool( False ),
    minDeltaPhi = cms.double( 0.0 ),
    maxDeltaPhi = cms.double( 2.0 ),
    minEtJet1 = cms.double( 20.0 ),
    minEtJet2 = cms.double( 20.0 ),
    Acoplanar = cms.string( "Jet1Jet2" )
)

