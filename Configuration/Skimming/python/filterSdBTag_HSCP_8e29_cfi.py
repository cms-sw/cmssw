import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterBTag_HSCP_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterBTag_HSCP_8e29.HLTPaths = ("HLT_BTagIP_Jet50","HLT_BTagMu_Jet10","HLT_StoppedHSCP")
filterBTag_HSCP_8e29.HLTPathsPrescales  = cms.vuint32(1,1,1)
filterBTag_HSCP_8e29.HLTOverallPrescale = cms.uint32(1)
filterBTag_HSCP_8e29.andOr = True
