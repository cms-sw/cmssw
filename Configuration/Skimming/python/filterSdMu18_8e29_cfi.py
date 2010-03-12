import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevelDev_cfi
SdMu9_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
SdMu9_8e29.HLTPaths = ("HLT_Mu9",)
SdMu9_8e29.HLTPathsPrescales  = cms.vuint32(1,)
SdMu9_8e29.HLTOverallPrescale = cms.uint32(1)
SdMu9_8e29.andOr = True

import HLTrigger.HLTfilters.hltSummaryFilter_cfi
HltSummaryMu18_8e29 = HLTrigger.HLTfilters.hltSummaryFilter_cfi.hltSummaryFilter.clone()
HltSummaryMu18_8e29.member = cms.InputTag("hltL3MuonCandidates","","HLT")
HltSummaryMu18_8e29.cut     = cms.string("pt>18")
HltSummaryMu18_8e29.minN    = cms.int32(1)     


filterSdMu18_8e29 = cms.Path(SdMu9_8e29*HltSummaryMu18_8e29)

