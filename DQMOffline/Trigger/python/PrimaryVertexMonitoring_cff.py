import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import pvMonitor

hltVerticesMonitoring = pvMonitor.clone(
    beamSpotLabel = "hltOnlineBeamSpot"
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltVerticesMonitoring,
                        TopFolderName = "HLT/Vertexing/hltFullVertices",
                        vertexLabel   = cms.InputTag("hltOfflinePrimaryVertices"))

hltPixelVerticesMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltPixelVertices",
    vertexLabel   = "hltPixelVertices",
    ndof          = 1,
    useHPforAlignmentPlots = False
)

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(hltPixelVerticesMonitoring,
                        vertexLabel = "hltPixelVerticesPPOnAA")

phase2_tracker.toModify(hltPixelVerticesMonitoring,
                        vertexLabel = "hltPhase2PixelVertices")

hltTrimmedPixelVerticesMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltTrimmedPixelVertices",
    vertexLabel   = "hltTrimmedPixelVertices",
    ndof          = 1,
    useHPforAlignmentPlots = False
)

pp_on_PbPb_run3.toModify(hltTrimmedPixelVerticesMonitoring,
                         vertexLabel = "hltTrimmedPixelVerticesPPOnAA")

hltVerticesPFFilterMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltVerticesPFFilter",
    vertexLabel   = "hltVerticesPFFilter",
    useHPforAlignmentPlots = False
)

pp_on_PbPb_run3.toModify(hltVerticesPFFilterMonitoring,
                         vertexLabel   = cms.InputTag("hltVerticesPFFilterPPOnAA"))

hltVerticesL3PFBjetsMonitoring = hltVerticesMonitoring.clone(
    TopFolderName = "HLT/Vertexing/hltVerticesL3PFBjets",
    vertexLabel   = "hltVerticesL3PFBjets",
    useHPforAlignmentPlots = False
)
vertexingMonitorHLT = cms.Sequence(
    hltPixelVerticesMonitoring +
    hltTrimmedPixelVerticesMonitoring +
    hltVerticesPFFilterMonitoring
)

phase2_tracker.toReplaceWith(vertexingMonitorHLT, cms.Sequence(hltPixelVerticesMonitoring + hltVerticesMonitoring))
