import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.MixingFamos_cff import *

#define the PU scenario itself
famosPileUp.PileUpSimulator = _block8TeV.PileUpSimulator
famosPileUp.PileUpSimulator.averageNumber = 0.000000

#also import the "no PU" option with the MixingModule:
from FastSimulation.PileUpProducer.mix_NoPileUp_cfi import *
