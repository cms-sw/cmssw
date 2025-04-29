import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
from Configuration.Eras.Modifier_run3_oxygen_cff import run3_oxygen

Run3_2025_OXY = cms.ModifierChain(Run3_2025, run3_upc, run3_oxygen)
