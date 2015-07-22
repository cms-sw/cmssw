import FWCore.ParameterSet.Config as cms

me0RecHits = cms.EDProducer("ME0RecHitProducer",
    recAlgoConfig = cms.PSet(),
    recAlgo = cms.string('ME0RecHitStandardAlgo'),
    me0DigiLabel = cms.InputTag("simMuonME0Digis"),
)
