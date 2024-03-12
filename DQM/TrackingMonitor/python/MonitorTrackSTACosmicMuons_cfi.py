import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackSTACosmicMuons = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/cosmicMuons',
    TrackProducer = 'cosmicMuons'
)

# foo bar baz
# PR4FYgQeyQ6FC
