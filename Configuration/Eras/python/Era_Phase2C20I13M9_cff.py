import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose

Phase2C20I13M9 = cms.ModifierChain(Phase2C17I13M9, phase2_hfnose)
