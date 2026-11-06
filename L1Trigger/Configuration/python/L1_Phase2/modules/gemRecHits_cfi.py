import FWCore.ParameterSet.Config as cms

gemRecHits = cms.EDProducer("GEMRecHitProducer",
    applyMasking = cms.bool(False),
    deadFile = cms.optional.FileInPath,
    ge21Container = cms.bool(False),
    gemDigiLabel = cms.InputTag("simMuonGEMDigis"),
    maskFile = cms.optional.FileInPath,
    mightGet = cms.optional.untracked.vstring,
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    recAlgoConfig = cms.PSet(

    )
)
