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

# Clone for TrackingMonitor for ZeroBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.andOr                = cms.bool( False )
TrackerCollisionTrackMonMB.dcsInputTag          = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMonMB.dcsPartitions        = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMonMB.andOrDcs             = cms.bool( False )
TrackerCollisionTrackMonMB.errorReplyDcs        = cms.bool( True )
TrackerCollisionTrackMonMB.dbLabel              = cms.string("SiStripDQMTrigger")
TrackerCollisionTrackMonMB.hltInputTag          = cms.InputTag( "TriggerResults::HLT" )
TrackerCollisionTrackMonMB.hltPaths             = cms.vstring("HLT_ZeroBias_v*")
TrackerCollisionTrackMonMB.hltDBKey             = cms.string("Tracker_MB")
TrackerCollisionTrackMonMB.errorReplyHlt        = cms.bool( False )
TrackerCollisionTrackMonMB.andOrHlt             = cms.bool(True)
TrackerCollisionTrackMonMB.doPrimaryVertexPlots = cms.bool(True)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

# Clone for TrackingMonitor for ZeroBias_IsolatedBunches ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonIsoBunches = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonIsoBunches.andOr                = cms.bool( False )
TrackerCollisionTrackMonIsoBunches.dcsInputTag          = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMonIsoBunches.dcsPartitions        = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMonIsoBunches.andOrDcs             = cms.bool( False )
TrackerCollisionTrackMonIsoBunches.errorReplyDcs        = cms.bool( True )
TrackerCollisionTrackMonIsoBunches.dbLabel              = cms.string("SiStripDQMTrigger")
TrackerCollisionTrackMonIsoBunches.hltInputTag          = cms.InputTag( "TriggerResults::HLT" )
TrackerCollisionTrackMonIsoBunches.hltPaths             = cms.vstring("HLT_ZeroBias_Isolated*")
TrackerCollisionTrackMonIsoBunches.hltDBKey             = cms.string("Tracker_MB")
TrackerCollisionTrackMonIsoBunches.errorReplyHlt        = cms.bool( False )
TrackerCollisionTrackMonIsoBunches.andOrHlt             = cms.bool(True)
TrackerCollisionTrackMonIsoBunches.doPrimaryVertexPlots = cms.bool(True)
TrackerCollisionTrackMonIsoBunches.setLabel("TrackerCollisionTrackMonIsoBunches")

