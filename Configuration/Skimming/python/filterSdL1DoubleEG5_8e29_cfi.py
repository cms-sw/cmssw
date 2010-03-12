import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
L1DoubleEG5_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
L1DoubleEG5_8e29.HLTPaths = ("HLT_L1DoubleEG5",)
L1DoubleEG5_8e29.HLTPathsPrescales  = cms.vuint32(5,)
L1DoubleEG5_8e29.HLTOverallPrescale = cms.uint32(1)
L1DoubleEG5_8e29.andOr = True

filterSdL1DoubleEG5_8e29 = cms.Path(L1DoubleEG5_8e29)
