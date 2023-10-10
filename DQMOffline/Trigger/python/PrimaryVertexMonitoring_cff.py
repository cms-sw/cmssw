import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import pvMonitor

hltVerticesMonitoring = pvMonitor.clone(
    beamSpotLabel = "hltOnlineBeamSpot"
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltVerticesMonitoring,
                        TopFolderName = "HLT/Vertexing/hltFullVertices",
                        vertexLabel   = cms.InputTag("offlinePrimaryVertices","","HLT"))

hltPixelVerticesMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltPixelVertices",
    vertexLabel   = "hltPixelVertices",
    ndof          = 1,
    useHPforAlignmentPlots = False
)

phase2_tracker.toModify(hltPixelVerticesMonitoring,
                        vertexLabel = "hltPhase2PixelVertices")

hltTrimmedPixelVerticesMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltTrimmedPixelVertices",
    vertexLabel   = "hltTrimmedPixelVertices",
    ndof          = 1,
    useHPforAlignmentPlots = False
)
hltVerticesPFFilterMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltVerticesPFFilter",
    vertexLabel   = "hltVerticesPFFilter",
    useHPforAlignmentPlots = False
)
hltVerticesL3PFBjetsMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltVerticesL3PFBjets",
    vertexLabel   = "hltVerticesL3PFBjets",
    useHPforAlignmentPlots = False
)
vertexingMonitorHLT = cms.Sequence(
    hltPixelVerticesMonitoring
    + hltTrimmedPixelVerticesMonitoring
    + hltVerticesPFFilterMonitoring
#    + hltVerticesL3PFBjets
)    

phase2_tracker.toReplaceWith(vertexingMonitorHLT, cms.Sequence(hltPixelVerticesMonitoring + hltVerticesMonitoring))
