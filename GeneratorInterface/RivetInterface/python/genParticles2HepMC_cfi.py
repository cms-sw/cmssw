import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("GenParticles2HepMCConverter",
    lheEvent = cms.InputTag("source"),
    genParticles = cms.InputTag("genParticles"),
#    genRunInfo   = cms.InputTag("generator"),
    genEventInfo = cms.InputTag("generator"),
)
