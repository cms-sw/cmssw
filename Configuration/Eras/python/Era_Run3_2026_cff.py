import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.Eras.Modifier_run3_l1scouting_2026_cff import run3_l1scouting_2026
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric

Run3_2026 = cms.ModifierChain(Run3_2025, PixelCPEGeneric, run3_l1scouting_2026)
