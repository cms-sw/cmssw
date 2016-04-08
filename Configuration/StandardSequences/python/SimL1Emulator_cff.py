import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.SimL1Emulator_cff import *

# make trigger digis available under with the raw2digi names
from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    # pretend these digis have been through digi2raw and to the HLT internal raw2digi, by using the approprate aliases
    # consider moving these mods to the HLT configuration
    from FastSimulation.Configuration.DigiAliases_cff import loadTriggerDigiAliases
    loadTriggerDigiAliases()
    from FastSimulation.Configuration.DigiAliases_cff import gtDigis,gmtDigis,gctDigis,caloStage1LegacyFormatDigis
    
#HGCal
def _modifyL1SimForHGCal( theProcess ):
    theProcess.simEcalTriggerPrimitiveDigis.BarrelOnly = cms.bool(True)  

from Configuration.StandardSequences.Eras import eras
modifyL1SimForHGCal_ = eras.phase2_hgcal.makeProcessModifier( _modifyL1SimForHGCal )



