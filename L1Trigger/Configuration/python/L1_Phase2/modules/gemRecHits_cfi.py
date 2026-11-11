import FWCore.ParameterSet.Config as cms

gemRecHits = cms.EDProducer("GEMRecHitProducer",
    applyMasking = cms.bool(False),
    ge21Container = cms.bool(False),
    gemDigiLabel = cms.InputTag("simMuonGEMDigis"),
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    recAlgoConfig = cms.PSet(

    )
)
