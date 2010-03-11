import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterMinBiasCal_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterMinBiasCal_8e29.HLTPaths = ("HLT_MinBiasEcal","HLT_MinBiasHcal")
filterMinBiasCal_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
filterMinBiasCal_8e29.HLTOverallPrescale = cms.uint32(1)
filterMinBiasCal_8e29.andOr = True
