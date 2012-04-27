from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import PileUpSimulatorBlock as block7TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = block7TeV.PileUpSimulator

famosPileUp.PileUpSimulator.averageNumber = 0.000000
