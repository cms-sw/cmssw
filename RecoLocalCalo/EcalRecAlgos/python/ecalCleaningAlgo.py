
import FWCore.ParameterSet.Config as cms


cleaningAlgoConfig = cms.PSet(
  # apply cleaning in EB above this threshold in GeV  
  cThreshold_barrel=cms.double(4),
  # apply cleaning in EE above this threshold in GeV 
  cThreshold_endcap=cms.double(15),
  # mark spike in EB if e4e1 <  e4e1_a_barrel_ * log10(e) + e4e1_b_barrel_
  e4e1_a_barrel=cms.double(0.02),
  e4e1_b_barrel=cms.double(0.02),
  # ditto for EE
  e4e1_a_endcap=cms.double(0.02),
  e4e1_b_endcap=cms.double(-0.0125),

  #when calculating e4/e1, ignore hits below this threshold
  e4e1Threshold_barrel= cms.double(0.080),
  e4e1Threshold_endcap= cms.double(0.300),
  
  # near cracks raise the energy threshold by this factor
  tightenCrack_e1_single=cms.double(1),
  # near cracks, divide the e4e1 threshold by this factor
  tightenCrack_e4e1_single=cms.double(2.5),
  # same as above for double spike
  tightenCrack_e1_double=cms.double(2),
  tightenCrack_e6e2_double=cms.double(3),
  # consider for double spikes if above this threshold
  cThreshold_double =cms.double(10),
  # mark double spike if e6e2< e6e2thresh
  e6e2thresh=cms.double(0.04),
  # ignore rechits flagged kOutOfTime above this energy threshold in EB
  ignoreOutOfTimeThresh=cms.double(1e9)
    )
