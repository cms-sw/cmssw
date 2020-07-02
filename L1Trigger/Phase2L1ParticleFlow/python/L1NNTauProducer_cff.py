import FWCore.ParameterSet.Config as cms

L1NNTauProducer = cms.EDProducer("L1NNTauProducer",
                                 seedpt          = cms.double(20),
                                 conesize        = cms.double(0.4),
                                 tausize         = cms.double(0.1),
                                 maxtaus         = cms.int32(5),
                                 nparticles      = cms.int32(10),
                                 L1PFObjects     = cms.InputTag("L1PFProducer","l1pfCandidates"),
                                 NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer.pb")
                                 )


L1NNTauProducerPuppi = cms.EDProducer("L1NNTauProducer",
                                 seedpt          = cms.double(20),
                                 conesize        = cms.double(0.4),
                                 tausize         = cms.double(0.1),
                                 maxtaus         = cms.int32(5),
                                 nparticles      = cms.int32(10),
                                 L1PFObjects     = cms.InputTag("L1PFProducer","l1pfCandidates"),
                                 NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb")
                                 )
