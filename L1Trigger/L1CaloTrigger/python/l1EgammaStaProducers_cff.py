import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *

L1TEgammaStaProducers = cms.Sequence(l1tEGammaClusterEmuProducer)

L1TEgammaStaProducersEB = cms.Sequence(l1tEGammaClusterEmuProducer)
