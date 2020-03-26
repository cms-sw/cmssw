import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.primaryVertexResolutionClient_cfi import primaryVertexResolutionClient as _primaryVertexResolutionClient

pixelVertexResolutionClient = _primaryVertexResolutionClient.clone(
    subDirs = ["OfflinePixelPV/Resolution/*"]
)
