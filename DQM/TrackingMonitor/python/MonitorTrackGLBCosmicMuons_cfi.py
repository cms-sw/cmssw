import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
MonitorTrackGLBCosmicMuons = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
MonitorTrackGLBCosmicMuons.TrackProducer = 'globalCosmicMuons'
MonitorTrackGLBCosmicMuons.AlgoName = 'glb'
MonitorTrackGLBCosmicMuons.FolderName = 'Muons/globalCosmicMuons'
MonitorTrackGLBCosmicMuons.doBeamSpotPlots = False
MonitorTrackGLBCosmicMuons.BSFolderName = 'Muons/globalCosmicMuons/BeamSpotParameters'
MonitorTrackGLBCosmicMuons.doAllSeedParameterHistos = False

