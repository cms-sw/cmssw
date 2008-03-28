import FWCore.ParameterSet.Config as cms

hltEgammaEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(26.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltEgammaL1MatchFilter")
)


