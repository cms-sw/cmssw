import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Mu9_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Mu9_8e29.HLTPaths = ("HLT_Mu9",)
Mu9_8e29.HLTPathsPrescales  = cms.vuint32(1,)
Mu9_8e29.HLTOverallPrescale = cms.uint32(1)
Mu9_8e29.andOr = True

filterSdMu9_8e29 = cms.Path(Mu9_8e29)
