import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterPhoton10_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterPhoton10_8e29.HLTPaths = ("HLT_Photon10_L1R",)
filterPhoton10_8e29.HLTPathsPrescales  = cms.vuint32(1,)
filterPhoton10_8e29.HLTOverallPrescale = cms.uint32(1)
filterPhoton10_8e29.andOr = True
