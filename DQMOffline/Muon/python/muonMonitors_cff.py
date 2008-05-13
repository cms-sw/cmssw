import FWCore.ParameterSet.Config as cms

#Analyzer taken from online dqm
from DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackGLBCosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackTKCosmicMuons_cfi import *
from DQM.DTMonitorModule.dtSegmTask_cfi import *
#dedicated analyzers for offline dqm 
from DQMOffline.Muon.muonAnalyzer_cff import *
from DQMOffline.Muon.CSCMonitor_cfi import *
import DQMOffline.Muon.muonAnalyzer_cfi
muonStandAloneAnalyzer = DQMOffline.Muon.muonAnalyzer_cfi.muonAnalyzer.clone()
#dedicated clients for offline dqm 
from DQMOffline.Muon.muonQualityTests_cff import *
muonTrackAnalyzers = cms.Sequence(MonitorTrackSTACosmicMuons*MonitorTrackTKCosmicMuons*MonitorTrackGLBCosmicMuons)
muonMonitors = cms.Sequence(muonTrackAnalyzers*dtSegmentsMonitor*cscMonitor*muonAnalyzer)
muonMonitors_woCSC = cms.Sequence(muonTrackAnalyzers*dtSegmentsMonitor*muonAnalyzer)
muonStandAloneMonitors = cms.Sequence(MonitorTrackSTACosmicMuons*dtSegmentsMonitor*cscMonitor*muonStandAloneAnalyzer)
muonMonitorsAndQualityTests = cms.Sequence(muonMonitors*muonQualityTests)
muonStandAloneAnalyzer.DoMuonRecoAnalysis = False

