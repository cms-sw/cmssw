import FWCore.ParameterSet.Config as cms

# dEdx monitor ####
#from DQM.TrackingMonitor.dEdxAnalyzer_cff import *
import DQM.TrackingMonitor.dEdxAnalyzer_cfi
# Clone for all PDs but ZeroBias ####
dEdxMonCommon = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()

dEdxHitMonCommon = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxHitAnalyzer.clone()

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
# Clone for ZeroBias ####
dEdxMonMB = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()
dEdxMonMB.dEdxParameters.andOr         = genericTriggerEventFlag4fullTrackerAndHLTdb.andOr
dEdxMonMB.dEdxParameters.dbLabel       = genericTriggerEventFlag4fullTrackerAndHLTdb.dbLabel
dEdxMonMB.dEdxParameters.andOrHlt      = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrHlt
dEdxMonMB.dEdxParameters.hltInputTag   = genericTriggerEventFlag4fullTrackerAndHLTdb.hltInputTag
dEdxMonMB.dEdxParameters.hltPaths      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltPaths
dEdxMonMB.dEdxParameters.hltDBKey      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltDBKey
dEdxMonMB.dEdxParameters.errorReplyHlt = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyHlt

dEdxHitMonMB = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxHitAnalyzer.clone()
dEdxHitMonMB.dEdxParameters.andOr         = genericTriggerEventFlag4fullTrackerAndHLTdb.andOr
dEdxHitMonMB.dEdxParameters.dbLabel       = genericTriggerEventFlag4fullTrackerAndHLTdb.dbLabel
dEdxHitMonMB.dEdxParameters.andOrHlt      = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrHlt
dEdxHitMonMB.dEdxParameters.hltInputTag   = genericTriggerEventFlag4fullTrackerAndHLTdb.hltInputTag
dEdxHitMonMB.dEdxParameters.hltPaths      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltPaths
dEdxHitMonMB.dEdxParameters.hltDBKey      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltDBKey
dEdxHitMonMB.dEdxParameters.errorReplyHlt = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyHlt

# Clone for SingleMu ####
dEdxMonMU = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()
dEdxMonMU.dEdxParameters.andOr         = cms.bool( False )
dEdxMonMU.dEdxParameters.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
dEdxMonMU.dEdxParameters.hltPaths      = cms.vstring("HLT_SingleMu40_Eta2p1_*")
dEdxMonMU.dEdxParameters.errorReplyHlt = cms.bool( False )
dEdxMonMU.dEdxParameters.andOrHlt      = cms.bool(True) 

dEdxHitMonMU = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxHitAnalyzer.clone()
dEdxHitMonMU.dEdxParameters.andOr         = cms.bool( False )
dEdxHitMonMU.dEdxParameters.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
dEdxHitMonMU.dEdxParameters.hltPaths      = cms.vstring("HLT_SingleMu40_Eta2p1_*")
dEdxHitMonMU.dEdxParameters.errorReplyHlt = cms.bool( False )
dEdxHitMonMU.dEdxParameters.andOrHlt      = cms.bool(True) 
