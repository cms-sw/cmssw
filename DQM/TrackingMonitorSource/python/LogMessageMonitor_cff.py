import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.LogMessageMonitor_cfi

# Clone for all PDs but MinBias ####
LogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonCommon.andOr         = cms.bool( False )
LogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
LogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
LogMessageMonCommon.andOrDcs      = cms.bool( False )
LogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for ZeroBias ###
LogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonMB.andOr         = cms.bool( False )
LogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
LogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
LogMessageMonMB.andOrDcs      = cms.bool( False )
LogMessageMonMB.errorReplyDcs = cms.bool( True )
LogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
LogMessageMonMB.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
LogMessageMonMB.hltPaths      = cms.vstring("HLT_ZeroBias_v*")
LogMessageMonMB.hltDBKey      = cms.string("Tracker_MB")
LogMessageMonMB.errorReplyHlt = cms.bool( False )
LogMessageMonMB.andOrHlt      = cms.bool(True) 

# Clone for ZeroBias_IsolatedBunches ###
LogMessageMonIsoBunches = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonIsoBunches.andOr         = cms.bool( False )
LogMessageMonIsoBunches.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
LogMessageMonIsoBunches.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
LogMessageMonIsoBunches.andOrDcs      = cms.bool( False )
LogMessageMonIsoBunches.errorReplyDcs = cms.bool( True )
LogMessageMonIsoBunches.dbLabel       = cms.string("SiStripDQMTrigger")
LogMessageMonIsoBunches.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
LogMessageMonIsoBunches.hltPaths      = cms.vstring("HLT_ZeroBias_Iso*")
LogMessageMonIsoBunches.hltDBKey      = cms.string("Tracker_MB")
LogMessageMonIsoBunches.errorReplyHlt = cms.bool( False )
LogMessageMonIsoBunches.andOrHlt      = cms.bool(True) 

