import FWCore.ParameterSet.Config as cms

me0RecHits = cms.EDProducer("ME0RecHitProducer",
    recAlgoConfig = cms.PSet(),
    recAlgo = cms.string('ME0RecHitStandardAlgo'),
    me0DigiLabel = cms.InputTag("simMuonME0PseudoReDigis"),
)
# foo bar baz
# 8JpvGw03cJU34
# vd8LUZPBY3wME
