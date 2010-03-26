import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Mu_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Mu_1e29.HLTPaths = ("HLT_L2Mu0","HLT_L2Mu3","HLT_L2Mu5")
Mu_1e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
Mu_1e29.HLTOverallPrescale = cms.uint32(1)
Mu_1e29.andOr = True

filterSdMu_1e29 = cms.Path(Mu_1e29)
