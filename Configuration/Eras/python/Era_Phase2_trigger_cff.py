import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2_cff import Phase2
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger

Phase2_trigger = cms.ModifierChain(Phase2, phase2_trigger)

