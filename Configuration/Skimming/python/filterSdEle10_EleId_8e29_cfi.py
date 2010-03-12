import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Ele10_EleId_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Ele10_EleId_8e29.HLTPaths = ("HLT_Ele10_LW_EleId_L1R",)
Ele10_EleId_8e29.HLTPathsPrescales  = cms.vuint32(1,)
Ele10_EleId_8e29.HLTOverallPrescale = cms.uint32(1)
Ele10_EleId_8e29.andOr = True

filterSdEle10_EleId_8e29 = cms.Path(Ele10_EleId_8e29)
