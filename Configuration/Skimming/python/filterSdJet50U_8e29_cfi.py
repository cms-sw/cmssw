import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterJet50U_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterJet50U_8e29.HLTPaths = ("HLT_Jet50U",)
filterJet50U_8e29.HLTPathsPrescales  = cms.vuint32(1,)
filterJet50U_8e29.HLTOverallPrescale = cms.uint32(1)
filterJet50U_8e29.andOr = True
