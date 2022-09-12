import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1tEGammaCrystalsEmulatorProducer_cfi import *

L1TEgammaStaProducers = cms.Sequence(l1tEGammaClusterEmuProducer)

L1TEgammaStaProducersEB = cms.Sequence(l1tEGammaClusterEmuProducer)
