import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
#from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexingTask = cms.Task(pixelTracksTask, pixelVertices)

from Configuration.ProcessModifiers.gpu_cff import gpu
_recopixelvertexingTask_gpu = recopixelvertexingTask.copy()

from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import pixelVertexCUDA
_recopixelvertexingTask_gpu.add(pixelVertexCUDA)

from RecoPixelVertexing.PixelVertexFinding.pixelVertexSoA_cfi import pixelVertexSoA
_recopixelvertexingTask_gpu.add(pixelVertexSoA)

from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA
# this is needed because the 'pixelTrack' EDAlias does not contain the 'ushorts' collections
_pixelVertexFromSoA.TrackCollection = 'pixelTrackFromSoA'
gpu.toModify(pixelVertices,
    cuda = _pixelVertexFromSoA
)

gpu.toReplaceWith(recopixelvertexingTask, _recopixelvertexingTask_gpu)

recopixelvertexing = cms.Sequence(recopixelvertexingTask)
