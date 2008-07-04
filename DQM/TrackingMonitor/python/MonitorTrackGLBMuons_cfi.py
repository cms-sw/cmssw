import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
MonitorTrackGLBMuons = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
MonitorTrackGLBMuons.TrackProducer = 'globalMuons'
MonitorTrackGLBMuons.AlgoName = 'glb'
MonitorTrackGLBMuons.FolderName = 'Muons/globalMuons'


