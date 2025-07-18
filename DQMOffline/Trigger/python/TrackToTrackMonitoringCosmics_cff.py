import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms =cms.untracked.bool(True)
from DQM.TrackingMonitorSource.TrackToTrackComparisonHists_cfi import TrackToTrackComparisonHists

hltCtfWithMaterialTracksP5_2_ctfWithMaterialTracksP5 = TrackToTrackComparisonHists.clone(
    monitoredTrack           = "hltCtfWithMaterialTracksP5",
    referenceTrack           = "ctfWithMaterialTracksP5",
    monitoredBeamSpot        = "hltOnlineBeamSpot",
    referenceBeamSpot        = "offlineBeamSpot",
    topDirName               = "HLT/Tracking/ValidationWRTOffline/hltCtfWithMaterialTracksP5",
    referencePrimaryVertices = "offlinePrimaryVertices",
    monitoredPrimaryVertices = "hltPixelVertices",
    isCosmics                = cms.bool(True),
    dxyCutForPlateau         = 1e6,
    histoPSet                = dict(
        Dxy_rangeMin = -60,
        Dxy_rangeMax = 60,
        Dxy_nbin = 120,
        Dz_rangeMin = -250,
        Dz_rangeMax =  250,
        Dz_nbin = 250,
    ) 
)

hltToOfflineCosmicsTrackValidatorSequence = cms.Sequence(hltCtfWithMaterialTracksP5_2_ctfWithMaterialTracksP5)
