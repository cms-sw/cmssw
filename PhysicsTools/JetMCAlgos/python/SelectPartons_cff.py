import FWCore.ParameterSet.Config as cms

# select the partons for Jet MC Flavour
myPartons = cms.EDProducer("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")
)


