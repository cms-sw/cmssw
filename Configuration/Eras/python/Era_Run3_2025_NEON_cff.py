import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_OXY_cff import Run3_2025_OXY
from Configuration.Eras.Modifier_run3_neon_cff import run3_neon

Run3_2025_NEON = cms.ModifierChain(Run3_2025_OXY, run3_neon)
