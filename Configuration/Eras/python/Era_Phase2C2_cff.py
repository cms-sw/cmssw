import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

Phase2C2 = cms.ModifierChain(run2_common, phase2_common, phase2_tracker, trackingPhase2PU140, phase2_hcal, phase2_hgcal, phase2_muon, run3_GEM)

