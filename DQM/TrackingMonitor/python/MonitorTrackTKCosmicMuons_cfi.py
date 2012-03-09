import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
MonitorTrackTKCosmicMuons = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
MonitorTrackTKCosmicMuons.TrackProducer = 'ctfWithMaterialTracksP5'
MonitorTrackTKCosmicMuons.AlgoName = 'ctf'
MonitorTrackTKCosmicMuons.FolderName = 'Muons/TKTrack'
MonitorTrackTKCosmicMuons.doBeamSpotPlots = False
MonitorTrackTKCosmicMuons.BSFolderName = 'Muons/TKTrack/BeamSpotParameters'
MonitorTrackTKCosmicMuons.doSeedParameterHistos = False
MonitorTrackTKCosmicMuons.doAllPlots = False
MonitorTrackTKCosmicMuons.doHitPropertiesPlots = True
MonitorTrackTKCosmicMuons.doGeneralPropertiesPlots = True


