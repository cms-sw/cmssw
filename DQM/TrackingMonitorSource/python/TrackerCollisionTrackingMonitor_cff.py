import FWCore.ParameterSet.Config as cms

# Clone for TrackingMonitor for all PDs but MinBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonCommon = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonCommon.andOr         = cms.bool( False )
TrackerCollisionTrackMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMonCommon.andOrDcs      = cms.bool( False )
TrackerCollisionTrackMonCommon.errorReplyDcs = cms.bool( True )
TrackerCollisionTrackMonCommon.setLabel("TrackerCollisionTrackMonCommon")

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
# Clone for TrackingMonitor for MinBias ###
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.andOr                = genericTriggerEventFlag4fullTrackerAndHLTdb.andOr
TrackerCollisionTrackMonMB.dcsInputTag          = genericTriggerEventFlag4fullTrackerAndHLTdb.dcsInputTag
TrackerCollisionTrackMonMB.dcsPartitions        = genericTriggerEventFlag4fullTrackerAndHLTdb.dcsPartitions
TrackerCollisionTrackMonMB.andOrDcs             = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrDcs
TrackerCollisionTrackMonMB.errorReplyDcs        = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyDcs
TrackerCollisionTrackMonMB.dbLabel              = genericTriggerEventFlag4fullTrackerAndHLTdb.dbLabel
TrackerCollisionTrackMonMB.andOrHlt             = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrHlt
TrackerCollisionTrackMonMB.hltInputTag          = genericTriggerEventFlag4fullTrackerAndHLTdb.hltInputTag
TrackerCollisionTrackMonMB.hltPaths             = genericTriggerEventFlag4fullTrackerAndHLTdb.hltPaths
TrackerCollisionTrackMonMB.hltDBKey             = genericTriggerEventFlag4fullTrackerAndHLTdb.hltDBKey
TrackerCollisionTrackMonMB.errorReplyHlt        = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyHlt
TrackerCollisionTrackMonMB.doPrimaryVertexPlots = cms.bool(True)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

