import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterZeroBias_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterZeroBias_8e29.HLTPaths = ("HLT_ZeroBias",)
filterZeroBias_8e29.HLTPathsPrescales  = cms.vuint32(12,)
filterZeroBias_8e29.HLTOverallPrescale = cms.uint32(1)
filterZeroBias_8e29.andOr = True
