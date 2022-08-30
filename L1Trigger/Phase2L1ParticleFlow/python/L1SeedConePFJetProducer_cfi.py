import FWCore.ParameterSet.Config as cms

l1tSeedConePFJetProducer = cms.EDProducer("L1SeedConePFJetProducer",
                           L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
                           nJets       = cms.uint32(10),
                           coneSize    = cms.double(0.4),
                           HW          = cms.bool(False),
                           debug       = cms.bool(False)
                         )

l1tSeedConePFJetEmulatorProducer = l1tSeedConePFJetProducer.clone(HW = True)

