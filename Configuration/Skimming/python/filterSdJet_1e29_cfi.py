import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Jet_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Jet_1e29.HLTPaths = ("HLT_L1Jet6U","HLT_Jet15U","HLT_Jet30U")
Jet_1e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
Jet_1e29.HLTOverallPrescale = cms.uint32(1)
Jet_1e29.andOr = True

filterSdJet_1e29 = cms.Path(Jet_1e29)
