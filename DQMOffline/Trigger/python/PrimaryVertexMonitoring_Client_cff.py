import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.primaryVertexResolutionClient_cfi import primaryVertexResolutionClient as _primaryVertexResolutionClient

hltPixelVertexResolutionClient = _primaryVertexResolutionClient.clone(
    subDirs = ["HLT/Vertexing/hltPixelVertices/Resolution/*"]
)

hltTrimmedPixelVertexResolutionClient = _primaryVertexResolutionClient.clone(
    subDirs = ["HLT/Vertexing/hltTrimmedPixelVertices/Resolution/*"]
)

hltFullVertexResolutionClient = _primaryVertexResolutionClient.clone(
    subDirs = ["HLT/Vertexing/hltVerticesPFFilter/Resolution/*"]
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltFullVertexResolutionClient,
                        subDirs = ["HLT/Vertexing/hltFullVertices/Resolution/*"])

hltVerticesMonitoringClient = cms.Sequence(hltPixelVertexResolutionClient+
                                           hltTrimmedPixelVertexResolutionClient+
                                           hltFullVertexResolutionClient)

phase2_tracker.toReplaceWith(hltVerticesMonitoringClient,
                             cms.Sequence(hltPixelVertexResolutionClient+
                                          hltFullVertexResolutionClient))
