import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_2024_L1T_cff import run3_2024_L1T
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_post2023_cff import run3_scouting_nanoAOD_post2023

Run3_2024 = cms.ModifierChain(Run3, run3_2024_L1T, run3_scouting_nanoAOD_post2023)
