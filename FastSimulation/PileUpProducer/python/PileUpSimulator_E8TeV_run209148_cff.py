from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import PileUpSimulatorBlock as _block8TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = _block8TeV.PileUpSimulator
famosPileUp.PileUpSimulator.usePoisson = False

famosPileUp.PileUpSimulator.probFunctionVariable = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20)
famosPileUp.PileUpSimulator.probValue = cms.vdouble(
                    1.92747e-08,
                    1.62702e-06,
                    7.42292e-05,
                    0.0017137,
                    0.0191414,
                    0.101638,
                    0.258023,
                    0.322184,
                    0.207559,
                    0.0730289,
                    0.0147525,
                    0.0017561,
                    0.000123411,
                    5.07434e-06,
                    1.20848e-07,
                    1.6531e-09,
                    1.29003e-11,
                    5.7105e-14,
                    1.42153e-16,
                    0,
                    0)

#also import the "no PU" option with the MixingModule:
from FastSimulation.PileUpProducer.mix_NoPileUp_cfi import *
