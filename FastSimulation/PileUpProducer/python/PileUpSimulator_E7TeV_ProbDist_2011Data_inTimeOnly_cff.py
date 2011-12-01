from FastSimulation.Configuration.RandomServiceInitialization_cff import *
from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *
from FastSimulation.Configuration.FamosSequences_cff import famosPileUp
famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator
famosPileUp.PileUpSimulator.usePoisson = False
famosPileUp.PileUpSimulator.probFunctionVariable = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34)
famosPileUp.PileUpSimulator.probValue = (0.000373819,0.000162138,0.002674,0.0092085,0.0695428,0.203376,0.291056,0.349973,0.356366,0.363203,0.405462,0.448033,0.488822,0.498761,0.450422,0.329016,0.188632,0.0840889,0.0309447,0.0105786,0.00334435,0.000915194,0.000207445,7.61739e-06,0,0,0,0,0,0,0,0,0,0,0)
