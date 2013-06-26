from FastSimulation.PileUpProducer.PileUpSimulator14TeV_cfi import PileUpSimulatorBlock as block14TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = block14TeV.PileUpSimulator
famosPileUp.PileUpSimulator.usePoisson = True
famosPileUp.PileUpSimulator.averageNumber = 50

