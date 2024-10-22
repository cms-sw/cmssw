import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackerCollisionTrackMon = TrackMon.clone(
    # Update specific parameters

    # input tags
    TrackProducer = "generalTracks",
    SeedProducer = "initialStepSeeds",
    TCProducer = "initialStepTrackCandidates",
    ClusterLabels = ('Tot','Strip','Pix',), # to decide which Seeds-Clusters correlation plots to have default is Total other options 'Strip', 'Pix'
    beamSpot = "offlineBeamSpot",
    primaryVertex = 'offlinePrimaryVertices',
    primaryVertexInputTags = ('offlinePrimaryVertices',),    
    selPrimaryVertexInputTags = ('goodOfflinePrimaryVertices',),
    pvLabels = ('offline',),

    # output parameters
    AlgoName = 'GenTk',
    Quality = '',
    FolderName = 'Tracking/GlobalParameters',
    BSFolderName = 'Tracking/ParametersVsBeamSpot',

    # determines where to evaluate track parameters
    # 'ImpactPoint'  --> evalutate at impact point
    MeasurementState = 'ImpactPoint',

    # which plots to do
    doAllPlots = False,
    doGoodTrackPlots = cms.bool(True),
    doTrackerSpecific = True,
    doHitPropertiesPlots = True,
    doGeneralPropertiesPlots = True,
    doBeamSpotPlots = True,
    doSeedParameterHistos = False,
    doRecHitVsPhiVsEtaPerTrack = True,
    doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(True),
    doLayersVsPhiVsEtaPerTrack = True,
    doGoodTrackLayersVsPhiVsEtaPerTrack = cms.bool(True),
    doPUmonitoring = False,
    doPlotsVsBXlumi = False,
    doPlotsVsGoodPVtx = True,
    doEffFromHitPatternVsPU = True,
    doEffFromHitPatternVsBX = True,
    doEffFromHitPatternVsLUMI = True,

    # LS analysis
    doLumiAnalysis = True,     
    doProfilesVsLS = True,

    doSeedNumberHisto = False,
    doSeedETAHisto = False,
    doSeedVsClusterHisto = False,

    # Number of Tracks per Event
    TkSizeBin = 600,
    TkSizeMax = 2999.5,
    TkSizeMin = -0.5,

    # chi2 dof
    Chi2NDFBin = 80,
    Chi2NDFMax = 79.5,
    Chi2NDFMin = -0.5,

    # Number of seeds per Event
    TkSeedSizeBin = 100,
    TkSeedSizeMax = 499.5,
    TkSeedSizeMin = -0.5,

    # Number of Track Cadidates per Event
    TCSizeBin = 100,
    TCSizeMax = 499.5,
    TCSizeMin = -0.5,

    GoodPVtx = TrackMon.GoodPVtx.clone(
        GoodPVtxBin = 60,
        GoodPVtxMin = 0.,
        GoodPVtxMax = 60.
    )
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(TrackerCollisionTrackMon, GoodPVtx=dict(GoodPVtxBin = 150, GoodPVtxMax = 150.))
run3_common.toModify(TrackerCollisionTrackMon, NTrkPVtx=dict(NTrkPVtxMax = 200.))
run3_common.toModify(TrackerCollisionTrackMon, NClusStrMax = 299999.5)
run3_common.toModify(TrackerCollisionTrackMon, NTrk2D=dict(NTrk2DBin = 100, NTrk2DMax = 5999.5))
run3_common.toModify(TrackerCollisionTrackMon, PVBin = 75, PVMax = 149.5)
run3_common.toModify(TrackerCollisionTrackMon, TkSizeMax = 5999.5)

