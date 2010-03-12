import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
MinBiasPixel_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
MinBiasPixel_8e29.HLTPaths = ("HLT_MinBiasPixel","HLT_MinBiasPixel_Trk5")
MinBiasPixel_8e29.HLTPathsPrescales  = cms.vuint32(4,4)
MinBiasPixel_8e29.HLTOverallPrescale = cms.uint32(1)
MinBiasPixel_8e29.andOr = True

filterSdMinBiasPixel_8e29 = cms.Path(MinBiasPixel_8e29)

