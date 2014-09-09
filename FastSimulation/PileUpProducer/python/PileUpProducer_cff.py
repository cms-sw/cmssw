import FWCore.ParameterSet.Config as cms

from FastSimulation.PileUpProducer.PileUpSimulator_cfi import *

famosPileUp = cms.EDProducer("PileUpProducer",
    # The conditions for pile-up event generation
    PileUpSimulatorBlock,
)
