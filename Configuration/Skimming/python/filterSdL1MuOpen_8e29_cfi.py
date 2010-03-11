import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterL1MuOpen_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterL1MuOpen_8e29.HLTPaths = ("HLT_L1MuOpen",)
filterL1MuOpen_8e29.HLTPathsPrescales  = cms.vuint32(10,)
filterL1MuOpen_8e29.HLTOverallPrescale = cms.uint32(1)
filterL1MuOpen_8e29.andOr = True
