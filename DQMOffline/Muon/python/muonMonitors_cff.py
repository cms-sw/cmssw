import FWCore.ParameterSet.Config as cms

#Analyzer taken from online dqm
from DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackGLBCosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackTKCosmicMuons_cfi import *
#dedicated analyzers for offline dqm 
from DQMOffline.Muon.muonAnalyzer_cff import *
#dedicated clients for offline dqm 
from DQMOffline.Muon.muonQualityTests_cff import *
muonTrackAnalyzers = cms.Sequence(MonitorTrackSTACosmicMuons*MonitorTrackTKCosmicMuons*MonitorTrackGLBCosmicMuons)
muonMonitors = cms.Sequence(muonTrackAnalyzers*muonAnalyzer)
muonMonitorsAndQualityTests = cms.Sequence(muonMonitors*muonQualityTests)

