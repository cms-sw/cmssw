import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger

Phase2C8_trigger = cms.ModifierChain(Phase2C8, phase2_trigger)

