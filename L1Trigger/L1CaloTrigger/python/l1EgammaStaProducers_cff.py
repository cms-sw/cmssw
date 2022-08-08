import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *

l1EgammaStaProducers = cms.Sequence(L1EGammaClusterEmuProducer)

l1EgammaStaProducersEB = cms.Sequence(L1EGammaClusterEmuProducer)
