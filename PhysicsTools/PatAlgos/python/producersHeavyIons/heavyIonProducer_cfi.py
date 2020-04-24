import FWCore.ParameterSet.Config as cms

heavyIon = cms.EDProducer("GenHIEventProducer",
  doReco     = cms.bool(True),
  doMC       = cms.bool(True),
  generators = cms.vstring("generatorSmeared")
)



