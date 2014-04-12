from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import PileUpSimulatorBlock as block7TeV
from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = block7TeV.PileUpSimulator
famosPileUp.PileUpSimulator.usePoisson = False
famosPileUp.PileUpSimulator.probFunctionVariable = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34)
famosPileUp.PileUpSimulator.probValue = (0.0109502,0.0263612,0.0456977,0.0641918,0.0781419,0.0861372,0.0886327,0.0869205,0.0823245,0.0758551,0.0681988,0.0598428,0.0511919,0.0426258,0.0344998,0.0271142,0.0206807,0.0153064,0.0109952,0.00766947,0.00519823,0.00342629,0.00219819,0.00137404,0.000837631,0.000498498,0.000289907,0.000164916,9.18504e-05,5.01304e-05,2.68344e-05,1.40992e-05,7.27646e-06,7.26061e-06,0)

#also import the "no PU" option with the MixingModule:
from FastSimulation.PileUpProducer.mix_NoPileUp_cfi import *
