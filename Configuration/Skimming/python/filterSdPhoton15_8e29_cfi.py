import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Photon15_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Photon15_8e29.HLTPaths = ("HLT_Photon15_L1R","HLT_Photon15_TrackIso_L1R","HLT_Photon15_LooseEcalIso_L1R")
Photon15_8e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
Photon15_8e29.HLTOverallPrescale = cms.uint32(1)
Photon15_8e29.andOr = True

filterSdPhoton15_8e29 = cms.Path(Photon15_8e29)
