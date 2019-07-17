import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C4_cff import Phase2C4
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger

Phase2C4_trigger = cms.ModifierChain(Phase2C4, phase2_trigger)

