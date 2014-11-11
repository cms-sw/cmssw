from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import PileUpSimulatorBlock as _block8TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = _block8TeV.PileUpSimulator
famosPileUp.PileUpSimulator.usePoisson = False

famosPileUp.PileUpSimulator.probFunctionVariable = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45)
famosPileUp.PileUpSimulator.probValue = cms.vdouble(
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0.0397281,
                           0.294024,
                           0.301409,
                           0.226158,
                           0.138681,
                           0)

#also import the "no PU" option with the MixingModule:
from FastSimulation.PileUpProducer.mix_NoPileUp_cfi import *
