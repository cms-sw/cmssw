import FWCore.ParameterSet.Config as cms

hltEgammaEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltEgammaL1MatchFilter" ),
    etcutEB = cms.double( 1.0 ),
    etcutEE = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    saveTags = cms.bool( False ),
    relaxed = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)


