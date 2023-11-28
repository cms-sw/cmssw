import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff import l1tLayer1Barrel,l1tLayer1HGCal,l1tLayer1

#from L1Trigger.Phase2L1ParticleFlow.L1NNTauProducer_cfi import *

#L1NNTauProducerPuppi = L1NNTauProducer.clone(
#                                NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb")
#                                )


l1tNNTauProducerPuppi = cms.EDProducer("L1NNTauProducer",
                                      seedpt          = cms.double(10),
                                      conesize        = cms.double(0.4),
                                      tausize         = cms.double(0.1),
                                      maxtaus         = cms.int32(5),
                                      nparticles      = cms.int32(10),
                                      HW              = cms.bool(True),
                                      emseed          = cms.bool(True),
                                      debug           = cms.bool(False),
                                      L1PFObjects     = cms.InputTag('l1tLayer2Deregionizer:Puppi'), #1pfCandidates:Puppi"),#l1pfCandidates
                                      NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb")
)

l1tNNTauProducerPF = cms.EDProducer("L1NNTauProducer",
                                      seedpt          = cms.double(10),
                                      conesize        = cms.double(0.4),
                                      tausize         = cms.double(0.1),
                                      maxtaus         = cms.int32(5),
                                      nparticles      = cms.int32(10),
                                      L1PFObjects     = cms.InputTag("l1tLayer1:PF"),#l1pfCandidates
                                      NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer.pb")
)

