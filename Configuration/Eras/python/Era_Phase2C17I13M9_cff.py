import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
from Configuration.Eras.Modifier_phase2_hgcalV12_cff import phase2_hgcalV12
from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16

Phase2C17I13M9 = cms.ModifierChain(Phase2C11I13M9.copyAndExclude([phase2_hgcalV12]),phase2_hgcalV16)
