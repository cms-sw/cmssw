import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
TrackerCosmics_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
TrackerCosmics_8e29.HLTPaths = ("HLT_TrackerCosmics",)
TrackerCosmics_8e29.HLTPathsPrescales  = cms.vuint32(1,)
TrackerCosmics_8e29.HLTOverallPrescale = cms.uint32(1)
TrackerCosmics_8e29.andOr = True

filterSdTrackerCosmics_8e29 = cms.Path(TrackerCosmics_8e29)
