import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Jet15U_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Jet15U_8e29.HLTPaths = ("HLT_Jet15U",)
Jet15U_8e29.HLTPathsPrescales  = cms.vuint32(10,)
Jet15U_8e29.HLTOverallPrescale = cms.uint32(1)
Jet15U_8e29.andOr = True

filterSdJet15U_8e29 = cms.Path(Jet15U_8e29)
