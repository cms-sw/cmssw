import FWCore.ParameterSet.Config as cms

# TrackingMonitor ####
# Clone for Cosmic Track Finder
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMon_cosmicTk = TrackerCosmicTrackMon.clone(
    TrackProducer = 'cosmictrackfinderP5',
    AlgoName = 'CosmicTk',
    FolderName = 'Tracking/TrackParameters',
    doSeedParameterHistos = True,
    TkSizeBin = 4,
    TkSizeMax = 3.5,
    TkSizeMin = -0.5
)

# Clone for CKF Tracks
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMon_ckf = TrackerCosmicTrackMon.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    AlgoName = 'CKFTk',
    FolderName = 'Tracking/TrackParameters',
    doSeedParameterHistos = True,
    TkSizeBin = 4,
    TkSizeMax = 3.5,
    TkSizeMin = -0.5
)

# Clone for Road Search  Tracks
# from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
# TrackMon_rs = TrackerCosmicTrackMon.clone(
#     TrackProducer = 'rsWithMaterialTracksP5',
#     AlgoName = 'RSTk',
#     FolderName = 'Tracking/TrackParameters',
#     doSeedParameterHistos = True
# )

# Clone for General Track (for Collision data)
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackMon_gentk = TrackerCollisionTrackMon.clone(
    FolderName = 'Tracking/TrackParameters',
    BSFolderName = 'Tracking/TrackParameters/BeamSpotParameters'
    # decrease number of histograms
    # doTrackerSpecific = False
)

# Clone for Heavy Ion Tracks (for HI Collisions)
from DQM.TrackingMonitor.TrackerHeavyIonTrackingMonitor_cfi import *
TrackMon_hi = TrackerHeavyIonTrackMon.clone(
    FolderName = 'Tracking/TrackParameters',
    BSFolderName = 'Tracking/TrackParameters/BeamSpotParameters'
)

# Tracking Efficiency ####
# Clone for Cosmic Tracks
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_cosmicTk = TrackEffMon.clone(
    TKTrackCollection = 'cosmictrackfinderP5',
    AlgoName = 'CosmicTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)

# Clone for CKF Tracks
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_ckf = TrackEffMon.clone(
    TKTrackCollection = 'ctfWithMaterialTracksP5',
    AlgoName = 'CKFTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)

# Clone for RS Tracks
# from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
# TrackEffMon_rs = TrackEffMon.clone(
#     TKTrackCollection = 'rsWithMaterialTracksP5',
#     AlgoName = 'RSTk',
#     FolderName = 'Tracking/TrackParameters/TrackEfficiency'
# )

# Clone for Beam Halo  Tracks
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_bhmuon = TrackEffMon.clone(
    TKTrackCollection = 'ctfWithMaterialTracksBeamHaloMuon',
    AlgoName = 'BHMuonTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)

# Clone for Heavy Ion Tracks (for HI Collisions)
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_hi = TrackEffMon.clone(
    TKTrackCollection = 'hiGeneralTracks',
    AlgoName = 'HeavyIonTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)
