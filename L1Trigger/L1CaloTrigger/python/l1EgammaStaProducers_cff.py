import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *

l1EgammaStaProducers = cms.Sequence(l1tEGammaClusterEmuProducer)

l1EgammaStaProducersEB = cms.Sequence(l1tEGammaClusterEmuProducer)
