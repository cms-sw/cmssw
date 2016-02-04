import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi
MonitorTrackSTACosmicMuonsHLTDT = DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi.MonitorTrackSTAMuons.clone()
import DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi
MonitorTrackSTACosmicMuonsHLTCSC = DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi.MonitorTrackSTAMuons.clone()
MonitorTrackSTACosmicMuonsHLTDT.TrackProducer = 'dtCosmicSTA'
MonitorTrackSTACosmicMuonsHLTDT.FolderName = 'Muons/cosmicMuonsHLTDT'
MonitorTrackSTACosmicMuonsHLTCSC.TrackProducer = 'cscCosmicSTA'
MonitorTrackSTACosmicMuonsHLTCSC.FolderName = 'Muons/cosmicMuonsHLTCSC'


