import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
pixelTracksMonitor = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
pixelTracksMonitor.FolderName                = 'Tracking/PixelTrackParameters/pixelTracks'
pixelTracksMonitor.TrackProducer             = 'pixelTracks'
pixelTracksMonitor.allTrackProducer          = 'pixelTracks'
pixelTracksMonitor.beamSpot                  = 'offlineBeamSpot'
pixelTracksMonitor.primaryVertex             = 'pixelVertices'
pixelTracksMonitor.pvNDOF                    = 1
pixelTracksMonitor.doAllPlots                = True
pixelTracksMonitor.doLumiAnalysis            = True
pixelTracksMonitor.doProfilesVsLS            = True
pixelTracksMonitor.doDCAPlots                = True
pixelTracksMonitor.doProfilesVsLS            = True
pixelTracksMonitor.doPlotsVsGoodPVtx         = True
pixelTracksMonitor.doEffFromHitPatternVsPU   = False
pixelTracksMonitor.doEffFromHitPatternVsBX   = False
pixelTracksMonitor.doEffFromHitPatternVsLUMI = False
pixelTracksMonitor.doPlotsVsGoodPVtx         = True
pixelTracksMonitor.doPlotsVsLUMI             = True
pixelTracksMonitor.doPlotsVsBX               = True

_trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('pixelTracks'),
    cut = cms.string("")
)

pixelTracksPt0to1 = _trackSelector.clone(cut = "pt >= 0 & pt < 1 ")
pixelTracksPt1 = _trackSelector.clone(cut = "pt >= 1 ")
from DQM.TrackingMonitorSource.TrackCollections2monitor_cff import highPurityPV0p1 as _highPurityPV0p1
pixelTracksPV0p1 = _highPurityPV0p1.clone(
    src = "pixelTracks",
    quality = "",
    vertexTag = "goodPixelVertices"
)

pixelTracksMonitorPt0to1 = pixelTracksMonitor.clone(
    TrackProducer = "pixelTracksPt0to1",
    FolderName = "Tracking/PixelTrackParameters/pt_0to1"
)
pixelTracksMonitorPt1 = pixelTracksMonitor.clone(
    TrackProducer = "pixelTracksPt1",
    FolderName = "Tracking/PixelTrackParameters/pt_1"
)
pixelTracksMonitorPV0p1 = pixelTracksMonitor.clone(
    TrackProducer = "pixelTracksPV0p1",
    FolderName = "Tracking/PixelTrackParameters/dzPV0p1"
)


from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices as _goodOfflinePrimaryVertices
goodPixelVertices = _goodOfflinePrimaryVertices.clone(
    src = "pixelVertices",
)

from DQM.TrackingMonitor.primaryVertexResolution_cfi import primaryVertexResolution as _primaryVertexResolution
pixelVertexResolution = _primaryVertexResolution.clone(
    vertexSrc = "goodPixelVertices",
    rootFolder = "OfflinePixelPV/Resolution",
)

pixelTracksMonitoringTask = cms.Task(
    goodPixelVertices,
    pixelTracksPt0to1,
    pixelTracksPt1,
    pixelTracksPV0p1,
)

pixelTracksMonitoring = cms.Sequence(
    pixelTracksMonitor +
    pixelTracksMonitorPt0to1 +
    pixelTracksMonitorPt1 +
    pixelTracksMonitorPV0p1 +
    pixelVertexResolution,
    pixelTracksMonitoringTask
)
