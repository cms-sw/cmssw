import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter

l1tGTQuadObjectCond = cms.EDFilter(
    "L1GTQuadObjectCond",
    scales=scale_parameter,
    # LUTs defined once in l1tGTLUTs_cfi, which must be loaded with this module
    cosh_eta_lut=cms.PSet(refToPSet_=cms.string("l1tGTCoshEtaLUT")),
    cosh_eta_lut2=cms.PSet(refToPSet_=cms.string("l1tGTCoshEtaLUT2")),
    cos_phi_lut=cms.PSet(refToPSet_=cms.string("l1tGTCosPhiLUT")),
    sanity_checks=cms.untracked.bool(False),
    inv_mass_checks=cms.untracked.bool(False),
    primVertTag = cms.InputTag("l1tGTProducer", "GTTPrimaryVert")
)
