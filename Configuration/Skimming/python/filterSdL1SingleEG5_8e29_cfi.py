import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterL1SingleEG5_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterL1SingleEG5_8e29.HLTPaths = ("HLT_L1SingleEG5","HLT_L1SingleEG8")
filterL1SingleEG5_8e29.HLTPathsPrescales  = cms.vuint32(5,5)
filterL1SingleEG5_8e29.HLTOverallPrescale = cms.uint32(1)
filterL1SingleEG5_8e29.andOr = True
