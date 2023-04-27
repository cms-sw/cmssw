import FWCore.ParameterSet.Config as cms

l1tSeedConePFJetProducer = cms.EDProducer("L1SeedConePFJetProducer")

l1tSeedConePFJetEmulatorProducer = l1tSeedConePFJetProducer.clone(HW = cms.bool(True))

