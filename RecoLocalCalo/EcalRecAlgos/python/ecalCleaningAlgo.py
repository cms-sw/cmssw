
import FWCore.ParameterSet.Config as cms


cleaningAlgoConfig = cms.PSet(
  cThreshold_barrel=cms.double(4), # apply cleaning above this threshold in GeV
  cThreshold_endcap=cms.double(15),  
  e4e1_a_barrel=cms.double(0.04),
  e4e1_b_barrel=cms.double(-0.024),
  e4e1_a_endcap=cms.double(0.02),
  e4e1_b_endcap=cms.double(-0.0125),
  cThreshold_double =cms.double(10),
  e4e1_IgnoreOutOfTime=cms.bool(True),
  tightenCrack_e1_single=cms.double(2),
  tightenCrack_e4e1_single=cms.double(3),
  tightenCrack_e1_double=cms.double(2),
  tightenCrack_e6e2_double=cms.double(3),
  e6e2thresh=cms.double(0.04),
    )
