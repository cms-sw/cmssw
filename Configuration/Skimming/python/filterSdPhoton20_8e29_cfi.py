import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterPhoton20_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterPhoton20_8e29.HLTPaths = ("HLT_Photon20_L1R","HLT_Photon30_L1R")
filterPhoton20_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
filterPhoton20_8e29.HLTOverallPrescale = cms.uint32(1)
filterPhoton20_8e29.andOr = True
