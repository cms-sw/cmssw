from FastSimulation.PileUpProducer.PileUpSimulator13TeV_cfi import PileUpSimulatorBlock as _block13TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = _block13TeV.PileUpSimulator
famosPileUp.PileUpSimulator.usePoisson = True
famosPileUp.PileUpSimulator.averageNumber = 20

