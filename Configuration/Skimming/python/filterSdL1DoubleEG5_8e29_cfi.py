import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterL1DoubleEG5_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterL1DoubleEG5_8e29.HLTPaths = ("HLT_L1DoubleEG5",)
filterL1DoubleEG5_8e29.HLTPathsPrescales  = cms.vuint32(5,)
filterL1DoubleEG5_8e29.HLTOverallPrescale = cms.uint32(1)
filterL1DoubleEG5_8e29.andOr = True
