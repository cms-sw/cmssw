import FWCore.ParameterSet.Config as cms

#Analyzer taken from online dqm
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackGLBMuons_cfi import *
from DQM.DTMonitorModule.dtSegmTask_cfi import *
#dedicated analyzers for offline dqm 
from DQMOffline.Muon.muonAnalyzer_cff import *
from DQMOffline.Muon.CSCMonitor_cfi import *
#dedicated clients for offline dqm 
from DQMOffline.Muon.muonQualityTests_cff import *
muonTrackAnalyzers = cms.Sequence(cms.SequencePlaceholder("standAloneMuonsMonitors")*MonitorTrackGLBMuons)
muonMonitors = cms.Sequence(muonTrackAnalyzers*dtSegmentsMonitor*cscMonitor*muonAnalyzer)
muonMonitorsAndQualityTests = cms.Sequence(muonMonitors*muonQualityTests)


