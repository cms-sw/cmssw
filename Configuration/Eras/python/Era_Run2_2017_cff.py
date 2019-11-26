import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
from Configuration.Eras.Modifier_run2_ECAL_2016_cff import run2_ECAL_2016
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
from Configuration.Eras.Modifier_run2_HLTconditions_2017_cff import run2_HLTconditions_2017
from Configuration.Eras.Modifier_run2_HLTconditions_2016_cff import run2_HLTconditions_2016
from Configuration.Eras.Modifier_run2_muon_2017_cff import run2_muon_2017
from Configuration.Eras.Modifier_run2_muon_2016_cff import run2_muon_2016
from Configuration.Eras.Modifier_run2_egamma_2017_cff import run2_egamma_2017
from Configuration.Eras.Modifier_run2_egamma_2016_cff import run2_egamma_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_pixel_2016_cff import pixel_2016
from Configuration.Eras.Modifier_run2_jme_2017_cff import run2_jme_2017
from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
from Configuration.Eras.Modifier_strips_vfp30_2016_cff import strips_vfp30_2016

Run2_2017 = cms.ModifierChain(Run2_2016.copyAndExclude([run2_muon_2016, run2_HLTconditions_2016, run2_ECAL_2016, run2_egamma_2016,pixel_2016,run2_jme_2016, strips_vfp30_2016]),
                              phase1Pixel, run2_ECAL_2017, run2_HF_2017, run2_HCAL_2017, run2_HE_2017, run2_HEPlan1_2017, 
                              trackingPhase1, run2_GEM_2017, stage2L1Trigger_2017, run2_HLTconditions_2017, run2_muon_2017,run2_egamma_2017, ctpps_2017, run2_jme_2017)

