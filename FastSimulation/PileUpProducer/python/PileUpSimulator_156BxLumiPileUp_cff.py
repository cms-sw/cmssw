from FastSimulation.Configuration.RandomServiceInitialization_cff import *
from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *
print "Simulated PileUp :  156BxLumiPileUp_cff"
from FastSimulation.Configuration.FamosSequences_cff import famosPileUp
famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator
famosPileUp.PileUpSimulator.averageNumber = 2.000000
