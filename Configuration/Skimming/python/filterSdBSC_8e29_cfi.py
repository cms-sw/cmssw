import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
BSC_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
BSC_8e29.HLTPaths = ("HLT_BackwardBSC","HLT_ForwardBSC")
BSC_8e29.HLTPathsPrescales  = cms.vuint32(1,1)
BSC_8e29.HLTOverallPrescale = cms.uint32(1)
BSC_8e29.andOr = True

filterSdBSC_8e29 = cms.Path(BSC_8e29)
