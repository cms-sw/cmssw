import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterL1_L2_Mu_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterL1_L2_Mu_8e29.HLTPaths = ("HLT_L2Mu9","HLT_L2Mu11","HLT_L1Mu20","HLT_L1Mu14_L1SingleEG10","HLT_L1Mu14_L1SingleJet6","HLT_L1Mu14_L1ETM30")
filterL1_L2_Mu_8e29.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1)
filterL1_L2_Mu_8e29.HLTOverallPrescale = cms.uint32(1)
filterL1_L2_Mu_8e29.andOr = True
