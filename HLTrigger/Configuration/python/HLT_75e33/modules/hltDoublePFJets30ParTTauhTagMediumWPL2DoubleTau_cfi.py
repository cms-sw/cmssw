import FWCore.ParameterSet.Config as cms

hltDoublePFJets30ParTTauhTagMediumWPL2DoubleTau = cms.EDFilter( "TauTagFilter",
    saveTags = cms.bool( True ),
    nExpected = cms.int32( 2 ),
    taus = cms.InputTag( "hltAK4PFPuppiJets" ),
    tauTags = cms.InputTag( 'hltParticleTransformerDiscriminatorsJetTags','TauvsAll' ),
    tauPtCorr = cms.InputTag( '','' ),
    seeds = cms.InputTag( "hltL1P2GTTau" ),
    seedTypes = cms.vint32( +84 ),
    selection = cms.string( "0.11645" ),
    minPt = cms.double( 30.0 ),
    maxEta = cms.double( 2.1 ),
    usePtCorr = cms.bool( False ),
    matchWithSeeds = cms.bool( False ),
    matchingdR = cms.double( 0.5 )
)
