import FWCore.ParameterSet.Config as cms

l1tNNTauProducerPuppi = cms.EDProducer("L1NNTauProducer",
    HW = cms.bool(True),
    L1PFObjects = cms.InputTag("l1tLayer2Deregionizer","Puppi"),
    NNFileName = cms.string('L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb'),
    conesize = cms.double(0.4),
    debug = cms.bool(False),
    emseed = cms.bool(True),
    maxtaus = cms.int32(5),
    nparticles = cms.int32(10),
    seedpt = cms.double(10),
    tauModelPath = cms.string('NNPuppiTauModel_v1'),
    tausize = cms.double(0.1)
)
