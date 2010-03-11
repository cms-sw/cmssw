import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterDoubleEG5_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterDoubleEG5_8e29.HLTPaths = ("HLT_DoubleEle5_SW_L1R","HLT_DoublePhoton5_eeRes_L1R")
filterDoubleEG5_8e29.HLTPathsPrescales  = cms.vuint32(1,20)
filterDoubleEG5_8e29.HLTOverallPrescale = cms.uint32(1)
filterDoubleEG5_8e29.andOr = True
