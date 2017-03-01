import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
MonitorTrackGLBMuons = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
MonitorTrackGLBMuons.TrackProducer = 'globalMuons'
MonitorTrackGLBMuons.AlgoName = 'glb'
MonitorTrackGLBMuons.FolderName = 'Muons/globalMuons'
MonitorTrackGLBMuons.doBeamSpotPlots = False
MonitorTrackGLBMuons.BSFolderName = 'Muons/globalCosmicMuons/BeamSpotParameters'
MonitorTrackGLBMuons.doSeedParameterHistos = False
MonitorTrackGLBMuons.doProfilesVsLS = False
MonitorTrackGLBMuons.doAllPlots = False
MonitorTrackGLBMuons.doGeneralPropertiesPlots = True
MonitorTrackGLBMuons.doHitPropertiesPlots = True
MonitorTrackGLBMuons.doTrackerSpecific = True
MonitorTrackGLBMuons.doDCAPlots = True
MonitorTrackGLBMuons.doDCAwrtPVPlots = True
MonitorTrackGLBMuons.doDCAwrt000Plots = False
MonitorTrackGLBMuons.doSIPPlots  = True
MonitorTrackGLBMuons.doEffFromHitPatternVsPU = False
MonitorTrackGLBMuons.doEffFromHitPatternVsBX = False

