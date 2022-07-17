import FWCore.ParameterSet.Config as cms

me0RecHits = cms.EDProducer("ME0RecHitProducer",
    me0DigiLabel = cms.InputTag("simMuonME0PseudoReDigis"),
    recAlgo = cms.string('ME0RecHitStandardAlgo'),
    recAlgoConfig = cms.PSet(

    )
)
