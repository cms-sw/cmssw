from IOMC.RandomEngine.IOMC_cff import *
from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *
from FastSimulation.Configuration.FamosSequences_cff import famosPileUp
famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator
famosPileUp.PileUpSimulator.averageNumber = 0.000000
