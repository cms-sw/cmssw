import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
#
# for STARTUP ONLY use try and use Offline 3D PV from pixelTracks, with adaptive vertex
#
from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
#from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
recopixelvertexingTask = cms.Task(pixelTracksTask,pixelVertices)
recopixelvertexing = cms.Sequence(recopixelvertexingTask)

from Configuration.ProcessModifiers.gpu_cff import gpu

from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import pixelVertexCUDA
from RecoPixelVertexing.PixelVertexFinding.pixelVertexSoA_cfi import pixelVertexSoA
from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA

_pixelVertexingCUDATask = cms.Task(pixelTracksTask,pixelVertexCUDA,pixelVertexSoA,pixelVertices)

# pixelVertexSoAonCPU = pixelVertexCUDA.clone()
# pixelVertexSoAonCPU.onGPU = False;

gpu.toReplaceWith(pixelVertices,_pixelVertexFromSoA)
gpu.toReplaceWith(recopixelvertexingTask,_pixelVertexingCUDATask)
