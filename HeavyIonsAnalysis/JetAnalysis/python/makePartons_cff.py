import FWCore.ParameterSet.Config as cms

myPartons = cms.EDProducer("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")
)

makePartons = cms.Sequence(myPartons)
