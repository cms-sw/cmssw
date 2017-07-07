import FWCore.ParameterSet.Config as cms

import SimMuon.MCTruth.MuonTrackProducer_cfi
muonInnerTrack = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
muonInnerTrack.muonsTag = cms.InputTag("muons")
muonInnerTrack.selectionTags = ('All',)
muonInnerTrack.trackType = "innerTrack"


import DQM.TrackingMonitor.TrackingMonitor_cfi
MonitorTrackMuonsInnerTrack = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
MonitorTrackMuonsInnerTrack.TrackProducer = 'muonInnerTrack'
MonitorTrackMuonsInnerTrack.AlgoName = 'inner'
MonitorTrackMuonsInnerTrack.FolderName = 'Muons/Tracking/innerTrack'
MonitorTrackMuonsInnerTrack.doBeamSpotPlots = True
MonitorTrackMuonsInnerTrack.BSFolderName = 'Muons/Tracking/innerTrack/BeamSpotParameters'
MonitorTrackMuonsInnerTrack.doSeedParameterHistos = False
MonitorTrackMuonsInnerTrack.doProfilesVsLS = False
MonitorTrackMuonsInnerTrack.doAllPlots = False
MonitorTrackMuonsInnerTrack.doGeneralPropertiesPlots = True
MonitorTrackMuonsInnerTrack.doHitPropertiesPlots = True
MonitorTrackMuonsInnerTrack.doTrackerSpecific = True
MonitorTrackMuonsInnerTrack.doDCAPlots = True
MonitorTrackMuonsInnerTrack.doDCAwrtPVPlots = True
MonitorTrackMuonsInnerTrack.doDCAwrt000Plots = False
MonitorTrackMuonsInnerTrack.doSIPPlots  = True
MonitorTrackMuonsInnerTrack.doEffFromHitPatternVsPU = True
MonitorTrackMuonsInnerTrack.doEffFromHitPatternVsBX = False

MonitorTrackINNMuons = cms.Sequence(muonInnerTrack+MonitorTrackMuonsInnerTrack)
