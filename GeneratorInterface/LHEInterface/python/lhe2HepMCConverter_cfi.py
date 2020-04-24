import FWCore.ParameterSet.Config as cms

lhe2HepMCConverter = cms.EDProducer("LHE2HepMCConverter",
  LHEEventProduct   = cms.InputTag("source"),
  LHERunInfoProduct = cms.InputTag("generatorSmeared")
)
