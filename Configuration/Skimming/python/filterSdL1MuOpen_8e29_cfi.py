import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
L1MuOpen_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
L1MuOpen_8e29.HLTPaths = ("HLT_L1MuOpen",)
L1MuOpen_8e29.HLTPathsPrescales  = cms.vuint32(10,)
L1MuOpen_8e29.HLTOverallPrescale = cms.uint32(1)
L1MuOpen_8e29.andOr = True

filterSdL1MuOpen_8e29 = cms.Path(L1MuOpen_8e29)
