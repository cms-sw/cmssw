import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.Eras.Modifier_run3_GEM_2025_cff import run3_GEM_2025
from Configuration.Eras.Modifier_run3_SiPixel_2025_cff import run3_SiPixel_2025

Run3_2025 = cms.ModifierChain(Run3_2024, run3_GEM_2025, run3_SiPixel_2025)
