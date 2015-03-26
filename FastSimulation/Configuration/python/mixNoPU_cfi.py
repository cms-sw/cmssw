#######################
# this file is the FastSim equivalent of SimGeneral/MixingModule/python/mixNoPU_cfi.py
# author: Lukas Vanelderen
# date:   Jan 20 2015
#######################

import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixNoPU_cfi import mix
from FastSimulation.Configuration.mixObjects_cfi import theMixObjects
from FastSimulation.Configuration.digitizers_cfi import *

mix.digitizers = theDigitizersValid # temporary: switch to theDigitizers as soon as there is a digi step for FastSim
mix.mixObjects = theMixObjects
