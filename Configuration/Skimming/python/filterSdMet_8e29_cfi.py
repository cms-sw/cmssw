import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
Met_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
Met_8e29.HLTPaths = ("HLT_L1MET20","HLT_MET35","HLT_MET100")
Met_8e29.HLTPathsPrescales  = cms.vuint32(10,1,1)
Met_8e29.HLTOverallPrescale = cms.uint32(1)
Met_8e29.andOr = True

filterSdMet_8e29 = cms.Path(Met_8e29)
