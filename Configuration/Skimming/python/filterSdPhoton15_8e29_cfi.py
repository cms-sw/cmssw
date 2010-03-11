import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterPhoton15_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterPhoton15_8e29.HLTPaths = ("HLT_Photon15_L1R","HLT_Photon15_TrackIso_L1R","HLT_Photon15_LooseEcalIso_L1R")
filterPhoton15_8e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
filterPhoton15_8e29.HLTOverallPrescale = cms.uint32(1)
filterPhoton15_8e29.andOr = True
