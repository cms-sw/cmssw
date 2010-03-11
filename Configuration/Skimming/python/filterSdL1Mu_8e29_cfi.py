import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterL1Mu_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterL1Mu_8e29.HLTPaths = ("HLT_L1Mu",)
filterL1Mu_8e29.HLTPathsPrescales  = cms.vuint32(2,)
filterL1Mu_8e29.HLTOverallPrescale = cms.uint32(1)
filterL1Mu_8e29.andOr = True
