from FastSimulation.Configuration.RandomServiceInitialization_cff import *
from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *
print "Simulated PileUp :  LowLumiPileUp_cff"
from FastSimulation.Configuration.FamosSequences_cff import famosPileUp
famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator
famosPileUp.PileUpSimulator.averageNumber = 7.100000
