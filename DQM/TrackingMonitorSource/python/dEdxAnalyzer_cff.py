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
dEdxMonMB.dEdxParameters.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb

dEdxHitMonMB = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxHitAnalyzer.clone()
dEdxHitMonMB.dEdxParameters.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb

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
