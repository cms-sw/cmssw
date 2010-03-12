import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Jet50U_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Jet50U_8e29.HLTPaths = ("HLT_Jet50U",)
Jet50U_8e29.HLTPathsPrescales  = cms.vuint32(1,)
Jet50U_8e29.HLTOverallPrescale = cms.uint32(1)
Jet50U_8e29.andOr = True

filterSdJet50U_8e29 = cms.Path(Jet50U_8e29)
