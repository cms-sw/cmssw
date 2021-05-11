import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *

# legacy pixel vertex reconsruction using the divisive vertex finder
pixelVerticesTask = cms.Task(
    pixelVertices
)

# "Patatrack" pixel ntuplets, fishbone cleaning, Broken Line fit, and density-based vertex reconstruction
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit

# build the pixel vertices in SoA format on the CPU
from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import pixelVertexCUDA as _pixelVertexCUDA
pixelVerticesSoA = SwitchProducerCUDA(
    cpu = _pixelVertexCUDA.clone(
        pixelTrackSrc = "pixelTracksSoA",
        onGPU = False
    )
)

# convert the pixel vertices from SoA to legacy format
from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA
pixelNtupletFit.toReplaceWith(pixelVertices, _pixelVertexFromSoA.clone(
    src = "pixelVerticesSoA"
))

pixelNtupletFit.toReplaceWith(pixelVerticesTask, cms.Task(
    # build the pixel vertices in SoA format on the CPU
    pixelVerticesSoA,
    # convert the pixel vertices from SoA to legacy format
    pixelVertices
))


# "Patatrack" sequence running on the GPU
from Configuration.ProcessModifiers.gpu_cff import gpu

# build pixel vertices in SoA format on the GPU
pixelVerticesCUDA = _pixelVertexCUDA.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    onGPU = True
)

# transfer the pixel vertices in SoA format to the CPU
from RecoPixelVertexing.PixelVertexFinding.pixelVertexSoA_cfi import pixelVertexSoA as _pixelVertexSoA
gpu.toModify(pixelVerticesSoA,
    cuda = _pixelVertexSoA.clone(
        src = cms.InputTag("pixelVerticesCUDA")
    )
)

(pixelNtupletFit & gpu).toReplaceWith(pixelVerticesTask, cms.Task(
    # build pixel vertices in SoA format on the GPU
    pixelVerticesCUDA,
    # transfer the pixel vertices in SoA format to the CPU and convert them to legacy format
    pixelVerticesTask.copy()
))

# Tasks and Sequences
recopixelvertexingTask = cms.Task(
    pixelTracksTask,
    pixelVerticesTask
)
recopixelvertexing = cms.Sequence(recopixelvertexingTask)
