import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_stage2L1Trigger_2024_cff import stage2L1Trigger_2024
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_2024_cff import run3_scouting_nanoAOD_2024

Run3_2024 = cms.ModifierChain(Run3, stage2L1Trigger_2024, run3_scouting_nanoAOD_2024)
