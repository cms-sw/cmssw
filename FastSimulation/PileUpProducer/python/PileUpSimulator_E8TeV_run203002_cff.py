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
                               9.66474e-17,
                               1.21114e-12,
                               2.94126e-09,
                               1.44281e-06,
                               0.000151792,
                               0.00376088,
                               0.0254935,
                               0.0610745,
                               0.0769054,
                               0.076596,
                               0.0737104,
                               0.0720484,
                               0.0702428,
                               0.065689,
                               0.0601684,
                               0.05682,
                               0.0557221,
                               0.0551615,
                               0.0541009,
                               0.051686,
                               0.0460924,
                               0.0368461,
                               0.0261355,
                               0.0164099,
                               0.0089456,
                               0.00410306,
                               0.00153858,
                               0.000462258,
                               0.000109812,
                               2.04474e-05,
                               2.96742e-06,
                               3.34444e-07,
                               2.9214e-08,
                               1.97586e-09,
                               1.03436e-10,
                               4.19123e-12,
                               1.31456e-13,
                               3.19116e-15,
                               5.99601e-17,
                               8.75296e-19,
                               0)
#also import the "no PU" option with the MixingModule:
from FastSimulation.PileUpProducer.mix_NoPileUp_cfi import *
