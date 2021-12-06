import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackerHeavyIonTrackMon = TrackMon.clone(
    # Update specific parameters
    TrackProducer = "hiGeneralTracks",
    SeedProducer = "hiPixelTrackSeeds",
    TCProducer = "hiPrimTrackCandidates",
    beamSpot = "offlineBeamSpot",
    primaryVertex = 'hiSelectedVertex',

    doHIPlots = True,

    AlgoName = 'HeavyIonTk',
    Quality = '',
    FolderName = 'Tracking/GlobalParameters',
    BSFolderName = 'Tracking/BeamSpotParameters',

    MeasurementState = 'ImpactPoint',

    doTrackerSpecific = True,
    doAllPlots = True,
    doBeamSpotPlots = True,
    doSeedParameterHistos = True,

    doLumiAnalysis = True,                       

    # Number of Tracks per Event
    TkSizeBin = 600,
    TkSizeMax = 1799.5,                        
    TkSizeMin = -0.5,

    # chi2 dof
    Chi2NDFBin = 160,
    Chi2NDFMax = 79.5,
    Chi2NDFMin = -0.5
)
