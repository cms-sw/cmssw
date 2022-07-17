import FWCore.ParameterSet.Config as cms

l1NNTauProducerPuppi = cms.EDProducer("L1NNTauProducer",
    L1PFObjects = cms.InputTag("l1pfCandidates","Puppi"),
    NNFileName = cms.string('L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb'),
    conesize = cms.double(0.4),
    maxtaus = cms.int32(5),
    mightGet = cms.optional.untracked.vstring,
    nparticles = cms.int32(10),
    seedpt = cms.double(20),
    tausize = cms.double(0.1)
)
