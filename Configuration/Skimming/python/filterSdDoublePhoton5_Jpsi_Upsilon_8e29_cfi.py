import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
DoublePhoton5_Jpsi_Upsilon_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
DoublePhoton5_Jpsi_Upsilon_8e29.HLTPaths = ("HLT_DoublePhoton5_Jpsi_L1R","HLT_DoublePhoton5_Upsilon_L1R")
DoublePhoton5_Jpsi_Upsilon_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
DoublePhoton5_Jpsi_Upsilon_8e29.HLTOverallPrescale = cms.uint32(1)
DoublePhoton5_Jpsi_Upsilon_8e29.andOr = True

filterSdDoublePhoton5_Jpsi_Upsilon_8e29 = cms.Path(DoublePhoton5_Jpsi_Upsilon_8e29)
