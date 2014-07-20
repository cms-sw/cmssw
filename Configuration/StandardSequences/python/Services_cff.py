# The following comments couldn't be translated into the new config version:

#

import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
# Random numbers initialization service
# pick it up directly
from IOMC.RandomEngine.IOMC_cff import *
#an "intermediate layer" remains, just in case somebody is using it...
# from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from DQMServices.Core.DQMStore_cfg import *

# This needs to be taken out as soon as reading SLHC11 files is no longer required.
# Added 05/Jul/2014 by Mark Grimes as a horrible temporary hack.
from SimTracker.SiPixelDigitizer.RemapDetIdService_cfi import RemapDetIdService
