import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterEle15_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterEle15_8e29.HLTPaths = ("HLT_Ele15_LW_L1R","HLT_Ele15_SC10_LW_L1R","HLT_Ele20_LW_L1R")
filterEle15_8e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
filterEle15_8e29.HLTOverallPrescale = cms.uint32(1)
filterEle15_8e29.andOr = True
