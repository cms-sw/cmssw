import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
CSCBeamHalo_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
CSCBeamHalo_8e29.HLTPaths = ("HLT_CSCBeamHalo","HLT_CSCBeamHaloOverlapRing1","HLT_CSCBeamHaloOverlapRing2","HLT_CSCBeamHaloRing2or3")
CSCBeamHalo_8e29.HLTPathsPrescales  = cms.vuint32(10,1,1,1)
CSCBeamHalo_8e29.HLTOverallPrescale = cms.uint32(1)
CSCBeamHalo_8e29.andOr = True

filterSdCSCBeamHalo_8e29 = cms.Path(CSCBeamHalo_8e29)

