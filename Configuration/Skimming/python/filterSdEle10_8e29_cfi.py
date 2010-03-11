import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterEle10_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterEle10_8e29.HLTPaths = ("HLT_Ele10_LW_L1R",)
filterEle10_8e29.HLTPathsPrescales  = cms.vuint32(10,)
filterEle10_8e29.HLTOverallPrescale = cms.uint32(1)
filterEle10_8e29.andOr = True
