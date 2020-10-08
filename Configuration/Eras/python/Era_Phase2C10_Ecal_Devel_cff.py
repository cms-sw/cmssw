import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose

Phase2C10_Ecal_Devel = cms.ModifierChain(Phase2C10,phase2_hfnose,phase2_ecal_devel)

