import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
from Configuration.Eras.Modifier_run3_oxygen_cff import run3_oxygen
from Configuration.Eras.Modifier_run3_neon_cff import run3_neon

Run3_2025_NEON = cms.ModifierChain(Run3_2025, run3_upc, run3_oxygen, run3_neon)
