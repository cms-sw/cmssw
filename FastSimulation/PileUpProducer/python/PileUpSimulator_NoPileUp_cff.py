from FastSimulation.Configuration.RandomServiceInitialization_cff import *
from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *
print "The pile up is taken from 7 TeV files. To switch to other files remove the inclusion of PileUpSimulator7TeV_cfi"
from FastSimulation.Configuration.FamosSequences_cff import famosPileUp
famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator
famosPileUp.PileUpSimulator.averageNumber = 0.000000
