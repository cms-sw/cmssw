import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import pvMonitor as _pvMonitor
pixelPVMonitor = _pvMonitor.clone(
    TopFolderName = "OfflinePixelPV",
    vertexLabel = "pixelVertices",
    ndof        = cms.int32( 1 )
)
