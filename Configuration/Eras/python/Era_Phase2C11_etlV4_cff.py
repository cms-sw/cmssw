import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
from Configuration.Eras.Modifier_phase2_etlV4_cff import phase2_etlV4

Phase2C11_etlV4 = cms.ModifierChain(Phase2C11, phase2_etlV4)
