import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.Eras.Modifier_phase2_ecal_cff import phase2_ecal
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_hcalHardcodeConditions_cff import hcalHardcodeConditions
from Configuration.Eras.Modifier_hcalSkipPacker_cff import hcalSkipPacker

Phase2 = cms.ModifierChain(run2_common, phase2_common, phase2_tracker, trackingPhase2PU140, phase2_ecal,
    run2_HE_2017, run2_HF_2017, run2_HCAL_2017, run3_HB, phase2_hcal, phase2_hgcal, phase2_muon, run3_GEM, stage2L1Trigger,
    hcalHardcodeConditions, hcalSkipPacker,
)
