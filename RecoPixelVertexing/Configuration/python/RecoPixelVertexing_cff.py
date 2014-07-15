import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexing = cms.Sequence(PixelLayerTriplets*pixelTracks*pixelVertices)

