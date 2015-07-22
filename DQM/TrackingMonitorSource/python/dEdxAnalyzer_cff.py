import FWCore.ParameterSet.Config as cms

# dEdx monitor ####
#from DQM.TrackingMonitor.dEdxAnalyzer_cff import *
import DQM.TrackingMonitor.dEdxAnalyzer_cfi
# Clone for all PDs but MinBias ####
dEdxMonCommon = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()

dEdxHitMonCommon = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxHitAnalyzer.clone()

# Clone for MinBias ####
dEdxMonMB = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()
dEdxMonMB.dEdxParameters.andOr         = cms.bool( False )
dEdxMonMB.dEdxParameters.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
dEdxMonMB.dEdxParameters.hltPaths      = cms.vstring("HLT_ZeroBias_*")
dEdxMonMB.dEdxParameters.hltDBKey      = cms.string("Tracker_MB")
dEdxMonMB.dEdxParameters.errorReplyHlt = cms.bool( False )
dEdxMonMB.dEdxParameters.andOrHlt      = cms.bool(True) 

dEdxHitMonMB = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxHitAnalyzer.clone()
dEdxHitMonMB.dEdxParameters.andOr         = cms.bool( False )
dEdxHitMonMB.dEdxParameters.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
dEdxHitMonMB.dEdxParameters.hltPaths      = cms.vstring("HLT_ZeroBias_*")
dEdxHitMonMB.dEdxParameters.hltDBKey      = cms.string("Tracker_MB")
dEdxHitMonMB.dEdxParameters.errorReplyHlt = cms.bool( False )
dEdxHitMonMB.dEdxParameters.andOrHlt      = cms.bool(True) 

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
