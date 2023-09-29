import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTSingleInOutLUT import COS_PHI_LUT, COSH_ETA_LUT, COSH_ETA_LUT_2
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter

l1tGTDoubleObjectCond = cms.EDFilter(
    "L1GTDoubleObjectCond",
    scales=scale_parameter,
    cosh_eta_lut=COSH_ETA_LUT.config(),
    cosh_eta_lut2=COSH_ETA_LUT_2.config(),
    cos_phi_lut=COS_PHI_LUT.config(),
    sanity_checks=cms.untracked.bool(False),
    inv_mass_checks=cms.untracked.bool(False)
)
