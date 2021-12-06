import FWCore.ParameterSet.Config as cms

# TrackingMonitor ####
from DQM.TrackingMonitor.TrackerHeavyIonTrackingMonitor_cfi import *
TrackMon_hi = TrackerHeavyIonTrackMon.clone(
    FolderName = 'Tracking/TrackParameters',
    BSFolderName = 'Tracking/TrackParameters/BeamSpotParameters',
    TrackProducer = "hiGeneralTracks"
)

TrackMonDQMTier0_hi = cms.Sequence(TrackMon_hi)
