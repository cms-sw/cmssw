import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.LogMessageMonitor_cfi

# Clone for all PDs but MinBias ####
LogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonCommon.andOr         = cms.bool( False )
LogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
LogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
LogMessageMonCommon.andOrDcs      = cms.bool( False )
LogMessageMonCommon.errorReplyDcs = cms.bool( True )

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
# Clone for MinBias ###
LogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonMB.andOr         = genericTriggerEventFlag4fullTrackerAndHLTdb.andOr
LogMessageMonMB.dcsInputTag   = genericTriggerEventFlag4fullTrackerAndHLTdb.dcsInputTag
LogMessageMonMB.dcsPartitions = genericTriggerEventFlag4fullTrackerAndHLTdb.dcsPartitions
LogMessageMonMB.andOrDcs      = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrDcs
LogMessageMonMB.errorReplyDcs = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyDcs
LogMessageMonMB.dbLabel       = genericTriggerEventFlag4fullTrackerAndHLTdb.dbLabel
LogMessageMonMB.andOrHlt      = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrHlt
LogMessageMonMB.hltInputTag   = genericTriggerEventFlag4fullTrackerAndHLTdb.hltInputTag
LogMessageMonMB.hltPaths      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltPaths
LogMessageMonMB.hltDBKey      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltDBKey
LogMessageMonMB.errorReplyHlt = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyHlt


