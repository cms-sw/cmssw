import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_post2023_cff import run3_scouting_nanoAOD_post2023

Run3_post2023 = cms.ModifierChain(Run3,run3_scouting_nanoAOD_post2023) 
