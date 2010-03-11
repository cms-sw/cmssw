import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterMinBiasPixel_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterMinBiasPixel_8e29.HLTPaths = ("HLT_MinBiasPixel","HLT_MinBiasPixel_Trk5")
filterMinBiasPixel_8e29.HLTPathsPrescales  = cms.vuint32(4,4)
filterMinBiasPixel_8e29.HLTOverallPrescale = cms.uint32(1)
filterMinBiasPixel_8e29.andOr = True
