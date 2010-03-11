import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterSdMet_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterSdMet_8e29.HLTPaths = ("HLT_L1MET20","HLT_MET35","HLT_MET100")
filterSdMet_8e29.HLTPathsPrescales  = cms.vuint32(10,1,1)
filterSdMet_8e29.HLTOverallPrescale = cms.uint32(1)
filterSdMet_8e29.andOr = True
