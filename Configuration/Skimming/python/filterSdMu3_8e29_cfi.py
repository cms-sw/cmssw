import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterMu3_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterMu3_8e29.HLTPaths = ("HLT_Mu3",)
filterMu3_8e29.HLTPathsPrescales  = cms.vuint32(10,)
filterMu3_8e29.HLTOverallPrescale = cms.uint32(1)
filterMu3_8e29.andOr = True
