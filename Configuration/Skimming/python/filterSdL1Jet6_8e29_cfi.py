import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
L1Jet6_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
L1Jet6_8e29.HLTPaths = ("HLT_L1Jet6",)
L1Jet6_8e29.HLTPathsPrescales  = cms.vuint32(10,)
L1Jet6_8e29.HLTOverallPrescale = cms.uint32(1)
L1Jet6_8e29.andOr = True

filterSdL1Jet6_8e29 = cms.Path(L1Jet6_8e29)
