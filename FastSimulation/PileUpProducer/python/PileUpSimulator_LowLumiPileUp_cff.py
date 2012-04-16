from IOMC.RandomEngine.IOMC_cff import *
from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *
from FastSimulation.Configuration.FamosSequences_cff import famosPileUp
famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator
famosPileUp.PileUpSimulator.averageNumber = 7.100000
