import FWCore.ParameterSet.Config as cms

slimmedJPTJets = cms.EDProducer("JPTJetSlimmer",
    src = cms.InputTag("JetPlusTrackZSPCorJetAntiKt4"),
    srcCalo = cms.InputTag("slimmedCaloJets"),
    cut = cms.string("pt>25 && abs(eta) < 2.2")
)
# foo bar baz
# GsN9TF957Nr5p
# c6z0i6RjexzHc
