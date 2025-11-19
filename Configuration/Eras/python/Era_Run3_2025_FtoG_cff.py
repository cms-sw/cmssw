import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.Eras.Modifier_run3_CSC_2025_cff import run3_CSC_2025_FtoG

Run3_2025_FtoG = cms.ModifierChain(Run3_2025, run3_CSC_2025_FtoG)
