import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.SimL1Emulator_cff import *

#If PreMixing, don't run these modules during first step
SimL1Emulator.remove(SimL1TCalorimeter)
SimL1Emulator.remove(SimL1TechnicalTriggers)
SimL1Emulator.remove(SimL1TGlobal)

# make trigger digis available under with the raw2digi names
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    # pretend these digis have been through digi2raw and to the HLT internal raw2digi, by using the approprate aliases
    # consider moving these mods to the HLT configuration
    from FastSimulation.Configuration.DigiAliases_cff import loadTriggerDigiAliases
    loadTriggerDigiAliases()
    from FastSimulation.Configuration.DigiAliases_cff import gtDigis,gmtDigis,gctDigis,caloStage1LegacyFormatDigis
    
