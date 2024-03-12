import FWCore.ParameterSet.Config as cms

hltGemRecHits = cms.EDProducer("GEMRecHitProducer",
    applyMasking = cms.bool(False),
    deadFile = cms.optional.FileInPath,
    gemDigiLabel = cms.InputTag("simMuonGEMDigis"),
    maskFile = cms.optional.FileInPath,
    mightGet = cms.optional.untracked.vstring,
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    recAlgoConfig = cms.PSet(

    )
)
# foo bar baz
# cAr8W5Iiu3pLJ
# XezGlMVE2Se5D
