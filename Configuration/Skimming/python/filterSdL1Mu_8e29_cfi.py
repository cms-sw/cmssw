import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
L1Mu_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
L1Mu_8e29.HLTPaths = ("HLT_L1Mu",)
L1Mu_8e29.HLTPathsPrescales  = cms.vuint32(2,)
L1Mu_8e29.HLTOverallPrescale = cms.uint32(1)
L1Mu_8e29.andOr = True

filterSdL1Mu_8e29 = cms.Path(L1Mu_8e29)
