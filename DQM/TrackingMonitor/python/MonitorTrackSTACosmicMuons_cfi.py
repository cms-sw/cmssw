import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi
MonitorTrackSTACosmicMuons = DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi.MonitorTrackSTAMuons.clone()
MonitorTrackSTACosmicMuons.TrackProducer = 'cosmicMuons'

