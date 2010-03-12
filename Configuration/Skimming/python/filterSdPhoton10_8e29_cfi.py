import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Photon10_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Photon10_8e29.HLTPaths = ("HLT_Photon10_L1R",)
Photon10_8e29.HLTPathsPrescales  = cms.vuint32(1,)
Photon10_8e29.HLTOverallPrescale = cms.uint32(1)
Photon10_8e29.andOr = True

filterSdPhoton10_8e29 = cms.Path(Photon10_8e29)
