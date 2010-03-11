import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterBSC_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterBSC_8e29.HLTPaths = ("HLT_BackwardBSC","HLT_ForwardBSC")
filterBSC_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
filterBSC_8e29.HLTOverallPrescale = cms.uint32(1)
filterBSC_8e29.andOr = True
