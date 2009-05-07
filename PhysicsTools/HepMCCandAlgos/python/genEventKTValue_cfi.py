import FWCore.ParameterSet.Config as cms

genEventKTValue = cms.EDProducer("GenEventKTValueProducer",
    src = cms.InputTag("genParticles")
)
