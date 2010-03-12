import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
DoubleMu_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
DoubleMu_8e29.HLTPaths = ("HLT_DoubleMu0","HLT_DoubleMu3","HLT_L1DoubleMuOpen")
DoubleMu_8e29.HLTPathsPrescales  = cms.vuint32(1,1,5)
DoubleMu_8e29.HLTOverallPrescale = cms.uint32(1)
DoubleMu_8e29.andOr = True

filterSdDoubleMu_8e29 = cms.Path(DoubleMu_8e29)
