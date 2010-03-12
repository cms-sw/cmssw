import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
DoublePhoton10_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
DoublePhoton10_8e29.HLTPaths = ("HLT_DoublePhoton10_L1R",)
DoublePhoton10_8e29.HLTPathsPrescales  = cms.vuint32(1,)
DoublePhoton10_8e29.HLTOverallPrescale = cms.uint32(1)
DoublePhoton10_8e29.andOr = True

filterSdDoublePhoton10_8e29 = cms.Path(DoublePhoton10_8e29)
