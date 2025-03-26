import FWCore.ParameterSet.Config as cms

#### TrackingMonitor ####
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMon_cosmicTkHLT = TrackerCosmicTrackMon.clone(
    TrackProducer = 'hltCtfWithMaterialTracksP5',
    AlgoName = 'CtfWithMaterialTracksP5',
    FolderName = 'HLT/Tracking/hltCtfWithMaterialTracksP5',
    doSeedParameterHistos = True
)

cosmicTrackingMonitorHLT = cms.Sequence(TrackMon_cosmicTkHLT)
