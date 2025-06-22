import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19

Phase2C26I13M9 = cms.ModifierChain(Phase2C22I13M9, phase2_hgcalV19)
