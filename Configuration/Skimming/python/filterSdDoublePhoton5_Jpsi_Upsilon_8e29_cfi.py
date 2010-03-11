import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterDoublePhoton5_Jpsi_Upsilon_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterDoublePhoton5_Jpsi_Upsilon_8e29.HLTPaths = ("HLT_DoublePhoton5_Jpsi_L1R","HLT_DoublePhoton5_Upsilon_L1R")
filterDoublePhoton5_Jpsi_Upsilon_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
filterDoublePhoton5_Jpsi_Upsilon_8e29.HLTOverallPrescale = cms.uint32(1)
filterDoublePhoton5_Jpsi_Upsilon_8e29.andOr = True
