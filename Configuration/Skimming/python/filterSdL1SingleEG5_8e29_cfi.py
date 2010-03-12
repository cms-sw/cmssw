import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
L1SingleEG5_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
L1SingleEG5_8e29.HLTPaths = ("HLT_L1SingleEG5","HLT_L1SingleEG8")
L1SingleEG5_8e29.HLTPathsPrescales  = cms.vuint32(5,5)
L1SingleEG5_8e29.HLTOverallPrescale = cms.uint32(1)
L1SingleEG5_8e29.andOr = True

filterSdL1SingleEG5_8e29 = cms.Path(L1SingleEG5_8e29)
