import FWCore.ParameterSet.Config as cms

hltMe0RecHits = cms.EDProducer("ME0RecHitProducer",
    me0DigiLabel = cms.InputTag("simMuonME0PseudoReDigis"),
    recAlgo = cms.string('ME0RecHitStandardAlgo'),
    recAlgoConfig = cms.PSet(

    )
)
# foo bar baz
# 4vltNtQ01ig4w
# AwF6NZ62RbxAX
