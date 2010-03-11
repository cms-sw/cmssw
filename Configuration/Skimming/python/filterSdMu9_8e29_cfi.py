import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterSdMu9_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterSdMu9_8e29.HLTPaths = ("HLT_Mu9",)
filterSdMu9_8e29.HLTPathsPrescales  = cms.vuint32(1,)
filterSdMu9_8e29.HLTOverallPrescale = cms.uint32(1)
filterSdMu9_8e29.andOr = True
