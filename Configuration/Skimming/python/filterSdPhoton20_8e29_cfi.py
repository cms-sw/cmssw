import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Photon20_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Photon20_8e29.HLTPaths = ("HLT_Photon20_L1R","HLT_Photon30_L1R")
Photon20_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
Photon20_8e29.HLTOverallPrescale = cms.uint32(1)
Photon20_8e29.andOr = True

filterSdPhoton20_8e29 = cms.Path(Photon20_8e29)
