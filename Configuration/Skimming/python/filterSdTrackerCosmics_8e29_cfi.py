import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterTrackerCosmics_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterTrackerCosmics_8e29.HLTPaths = ("HLT_TrackerCosmics",)
filterTrackerCosmics_8e29.HLTPathsPrescales  = cms.vuint32(1,)
filterTrackerCosmics_8e29.HLTOverallPrescale = cms.uint32(1)
filterTrackerCosmics_8e29.andOr = True
