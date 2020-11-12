import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1EGammaEEProducer_cfi import *
from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *

l1EgammaStaProducers = cms.Sequence(l1EGammaEEProducer+L1EGammaClusterEmuProducer)

l1EgammaStaProducersEE = cms.Sequence(l1EGammaEEProducer)
l1EgammaStaProducersEB = cms.Sequence(L1EGammaClusterEmuProducer)
