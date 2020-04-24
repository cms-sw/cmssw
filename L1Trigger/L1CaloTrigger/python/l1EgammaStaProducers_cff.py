import FWCore.ParameterSet.Config as cms

# from L1Trigger.L1CaloTrigger.l1EGammaCrystalsProducer_cfi import *
from L1Trigger.L1CaloTrigger.l1EGammaEEProducer_cfi import *

l1EgammaStaProducers = cms.Sequence(l1EGammaEEProducer)
