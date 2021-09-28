import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackerCosmicTrackMon = TrackMon.clone(
    # Update specific parameters
    SeedProducer = "combinedP5SeedsForCTF",
    TCProducer = "ckfTrackCandidatesP5",
    beamSpot = "offlineBeamSpot",     

    MeasurementState = 'default',

    doAllPlots = False,
    doHitPropertiesPlots = True,
    doGeneralPropertiesPlots = True,
    doBeamSpotPlots = False,
    doSeedParameterHistos = False,

    Chi2Max = 500.0,

    TkSizeBin = 25,
    TkSizeMax = 24.5,

    TkSeedSizeBin = 20,
    TkSeedSizeMax = 19.5,

    RecLayBin = 35,
    RecLayMax = 34.5,

    TrackPtMax = 30.0,
    TrackPtMin = -0.5,

    TrackPxMax = 50.0,
    TrackPxMin = -50.0,

    TrackPyMax = 50.0,
    TrackPyMin = -50.0,

    TrackPzMax = 50.0,
    TrackPzMin = -50.0,

    doLumiAnalysis = False
)
