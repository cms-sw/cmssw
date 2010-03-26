import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
EGamma_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
EGamma_1e29.HLTPaths = ("HLT_L1SingleEG5","HLT_L1SingleEG8")
EGamma_1e29.HLTPathsPrescales  = cms.vuint32(1,1)
EGamma_1e29.HLTOverallPrescale = cms.uint32(1)
EGamma_1e29.andOr = True

filterSdEGamma_1e29 = cms.Path(EGamma_1e29)
