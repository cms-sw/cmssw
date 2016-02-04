import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from DQMOffline.Muon.muonAnalyzer_cfi import *
import DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cfi
staMonitor = DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cfi.MonitorTrackSTACosmicMuons.clone()
import DQM.TrackingMonitor.MonitorTrackGLBCosmicMuons_cfi
glbMonitor = DQM.TrackingMonitor.MonitorTrackGLBCosmicMuons_cfi.MonitorTrackGLBCosmicMuons.clone()
import DQM.TrackingMonitor.MonitorTrackTKCosmicMuons_cfi
tkMonitor = DQM.TrackingMonitor.MonitorTrackTKCosmicMuons_cfi.MonitorTrackTKCosmicMuons.clone()
from DQMOffline.JetMET.jetMETAnalyzer_cff import *
lhcMuonsMonitors = cms.Sequence(staMonitor*glbMonitor*tkMonitor*muonAnalyzer)
DQMOffline = cms.Sequence(lhcMuonsMonitors*jetMETAnalyzer)
staMonitor.TrackProducer = 'standAloneMuons'
glbMonitor.TrackProducer = 'globalMuons'
tkMonitor.TrackProducer = 'generalTracks'

