import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.SimL1Emulator_cff import *

# make trigger digis available under with the raw2digi names
from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    from FastSimulation.Configuration.DigiAliases import loadTriggerDigiAliases
    loadTriggerDigiAliases()
    from FastSimulation.Configuration.DigiAliases import gtDigis,gmtDigis
    
