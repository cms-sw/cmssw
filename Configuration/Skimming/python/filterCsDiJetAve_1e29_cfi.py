import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
DiJetAve_1e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
DiJetAve_1e29.HLTPaths = ("HLT_DiJetAve15U_8E29","HLT_DiJetAve30U_8E29")
DiJetAve_1e29.HLTPathsPrescales  = cms.vuint32(1,1)
DiJetAve_1e29.HLTOverallPrescale = cms.uint32(1)
DiJetAve_1e29.andOr = True

filterCsDiJetAve_1e29 = cms.Path(DiJetAve_1e29)
