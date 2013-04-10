from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import PileUpSimulatorBlock as _block8TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = _block8TeV.PileUpSimulator

famosPileUp.PileUpSimulator.averageNumber = 7.100000

#also import the "no PU" option with the MixingModule:
from FastSimulation.PileUpProducer.mix_NoPileUp_cfi import *
