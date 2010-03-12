import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
DoubleEG5_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
DoubleEG5_8e29.HLTPaths = ("HLT_DoubleEle5_SW_L1R","HLT_DoublePhoton5_eeRes_L1R")
DoubleEG5_8e29.HLTPathsPrescales  = cms.vuint32(1,20)
DoubleEG5_8e29.HLTOverallPrescale = cms.uint32(1)
DoubleEG5_8e29.andOr = True

filterSdDoubleEG5_8e29 = cms.Path(DoubleEG5_8e29)
