import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterJet15U_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterJet15U_8e29.HLTPaths = ("HLT_Jet15U",)
filterJet15U_8e29.HLTPathsPrescales  = cms.vuint32(10,)
filterJet15U_8e29.HLTOverallPrescale = cms.uint32(1)
filterJet15U_8e29.andOr = True
