import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
BTag_HSCP_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
BTag_HSCP_8e29.HLTPaths = ("HLT_BTagIP_Jet50","HLT_BTagMu_Jet10","HLT_StoppedHSCP")
BTag_HSCP_8e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
BTag_HSCP_8e29.HLTOverallPrescale = cms.uint32(1)
BTag_HSCP_8e29.andOr = True

filterSdBTag_HSCP_8e29 = cms.Path(BTag_HSCP_8e29)
