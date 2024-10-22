import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_noMkFit_cff import Run3_noMkFit
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.Eras.Modifier_phase2_ecal_cff import phase2_ecal
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.ProcessModifiers.seedingDeepCore_cff import seedingDeepCore
from Configuration.ProcessModifiers.displacedRegionalTracking_cff import displacedRegionalTracking
from Configuration.Eras.Modifier_hcalHardcodeConditions_cff import hcalHardcodeConditions
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
from Configuration.Eras.Modifier_ctpps_2022_cff import ctpps_2022
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Phase2 = cms.ModifierChain(Run3_noMkFit.copyAndExclude([phase1Pixel,trackingPhase1,seedingDeepCore,displacedRegionalTracking,ctpps_2022,dd4hep]), 
                           phase2_common, phase2_tracker, trackingPhase2PU140, phase2_ecal, phase2_hcal, phase2_hgcal, phase2_muon, phase2_GEM, hcalHardcodeConditions, phase2_timing, phase2_timing_layer, phase2_trigger)
