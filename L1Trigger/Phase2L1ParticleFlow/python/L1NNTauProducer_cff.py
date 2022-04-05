import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.L1NNTauProducer_cfi import *

L1NNTauProducerPuppi = L1NNTauProducer.clone(
                                NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb")
                                )
