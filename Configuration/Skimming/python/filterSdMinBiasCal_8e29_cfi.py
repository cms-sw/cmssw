import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
MinBiasCal_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
MinBiasCal_8e29.HLTPaths = ("HLT_MinBiasEcal","HLT_MinBiasHcal")
MinBiasCal_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
MinBiasCal_8e29.HLTOverallPrescale = cms.uint32(1)
MinBiasCal_8e29.andOr = True

filterSdMinBiasCal_8e29 = cms.Path(MinBiasCal_8e29)
