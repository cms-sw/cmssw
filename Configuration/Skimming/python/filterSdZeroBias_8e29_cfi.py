import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
ZeroBias_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
ZeroBias_8e29.HLTPaths = ("HLT_ZeroBias",)
ZeroBias_8e29.HLTPathsPrescales  = cms.vuint32(12,)
ZeroBias_8e29.HLTOverallPrescale = cms.uint32(1)
ZeroBias_8e29.andOr = True

filterSdZeroBias_8e29 = cms.Path(ZeroBias_8e29)
