# The following comments couldn't be translated into the new config version:

#

import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
# Random numbers initialization service
# pick it up directly
from IOMC.RandomEngine.IOMC_cff import *
#an "intermediate layer" remains, just in case somebody is using it...
# from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
DQMStore = cms.Service("DQMStore")



