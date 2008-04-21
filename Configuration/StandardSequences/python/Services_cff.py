# The following comments couldn't be translated into the new config version:

#

import FWCore.ParameterSet.Config as cms

#
#service = Timing { } 
#service = SimpleMemoryCheck {
#    untracked int32 ignoreTotal = 1 # default is one
#} 
#service = Tracer {} 
#
# not for the moment
#include "FWCore/Services/data/EnableFloatingPointExceptions.cfi"
from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
DQMStore = cms.Service("DQMStore")


