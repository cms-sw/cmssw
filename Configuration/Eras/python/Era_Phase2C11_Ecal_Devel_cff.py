import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel

Phase2C11_Ecal_Devel = cms.ModifierChain(Phase2C11,phase2_ecal_devel)

