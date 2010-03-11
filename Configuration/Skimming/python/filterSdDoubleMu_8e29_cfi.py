import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterDoubleMu_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterDoubleMu_8e29.HLTPaths = ("HLT_DoubleMu0","HLT_DoubleMu3","HLT_L1DoubleMuOpen")
filterDoubleMu_8e29.HLTPathsPrescales  = cms.vuint32(1,1,5)
filterDoubleMu_8e29.HLTOverallPrescale = cms.uint32(1)
filterDoubleMu_8e29.andOr = True
