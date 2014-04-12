import FWCore.ParameterSet.Config as cms

#Analyzer taken from online dqm
from DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cff import *
from DQM.TrackingMonitor.MonitorTrackSTACosmicMuonsHLT_cff import *
from DQM.TrackingMonitor.MonitorTrackGLBCosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackTKCosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackEfficiencySTACosmicMuons_cff import *
from DQM.TrackingMonitor.MonitorTrackEfficiencyTkTracks_cff import *
from DQMOffline.Muon.dtSegmTask_cfi import *

#dedicated analyzers for offline dqm
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from DQMOffline.Muon.muonCosmicAnalyzer_cff import *
from DQMOffline.Muon.CSCMonitor_cfi import *
from DQMOffline.Muon.muonIdDQM_cff import *

#dedicated clients for offline dqm
from DQMOffline.Muon.muonQualityTests_cff import *


dqmInfoMuons = cms.EDAnalyzer("DQMEventInfo",
                              subSystemFolder = cms.untracked.string('Muons')
                              )

muonTrackCosmicAnalyzers = cms.Sequence(standAloneCosmicMuonsMonitors*
                                        MonitorTrackTKCosmicMuons*
                                        MonitorTrackGLBCosmicMuons*
                                        MonitorTrackEfficiencySTACosmicMuons*
                                        MonitorTrackEfficiencyTkTracks)

muonTrackCosmicAnalyzersHLT = cms.Sequence(MonitorTrackSTACosmicMuonsHLTDT*
                                           MonitorTrackSTACosmicMuonsHLTCSC)

muonCosmicMonitors = cms.Sequence(muonTrackCosmicAnalyzers*
                                  dtSegmentsMonitor*
                                  cscMonitor*
                                  muonCosmicAnalyzer*
                                  muonIdDQM*
                                  dqmInfoMuons)

##muonCosmicMonitors = cms.Sequence(muonTrackCosmicAnalyzers*dtSegmentsMonitor*cscMonitor*muonCosmicAnalyzer)

muonCosmicMonitors_woCSC = cms.Sequence(cms.SequencePlaceholder("muonTrackAnalyzers")*
                                        dtSegmentsMonitor*
                                        muonCosmicMonitors)

muonStandAloneCosmicMonitors = cms.Sequence(MonitorTrackSTACosmicMuons*
                                            dtSegmentsMonitor*
                                            cscMonitor*
                                            muonSACosmicAnalyzer)

muonCosmicMonitorsAndQualityTests = cms.Sequence(muonCosmicMonitors*muonQualityTests)

