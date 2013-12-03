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

# Clone for TrackingMonitor for MinBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.andOr                = cms.bool( False )
TrackerCollisionTrackMonMB.dcsInputTag          = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMonMB.dcsPartitions        = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMonMB.andOrDcs             = cms.bool( False )
TrackerCollisionTrackMonMB.errorReplyDcs        = cms.bool( True )
TrackerCollisionTrackMonMB.dbLabel              = cms.string("SiStripDQMTrigger")
TrackerCollisionTrackMonMB.hltInputTag          = cms.InputTag( "TriggerResults::HLT" )
TrackerCollisionTrackMonMB.hltPaths             = cms.vstring("HLT_ZeroBias_*")
TrackerCollisionTrackMonMB.hltDBKey             = cms.string("Tracker_MB")
TrackerCollisionTrackMonMB.errorReplyHlt        = cms.bool( False )
TrackerCollisionTrackMonMB.andOrHlt             = cms.bool(True)
TrackerCollisionTrackMonMB.doPrimaryVertexPlots = cms.bool(True)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

