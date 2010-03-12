import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Mu3_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Mu3_8e29.HLTPaths = ("HLT_Mu3",)
Mu3_8e29.HLTPathsPrescales  = cms.vuint32(10,)
Mu3_8e29.HLTOverallPrescale = cms.uint32(1)
Mu3_8e29.andOr = True

filterSdMu3_8e29 = cms.Path(Mu3_8e29)
