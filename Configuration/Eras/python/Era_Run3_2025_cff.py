import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_2025_cff import run3_scouting_nanoAOD_2025

Run3_2025 = cms.ModifierChain(Run3_2024, run3_scouting_nanoAOD_2025)
