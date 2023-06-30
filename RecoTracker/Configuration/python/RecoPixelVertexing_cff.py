import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from RecoTracker.PixelTrackFitting.PixelTracks_cff import *
from RecoTracker.PixelVertexFinding.PixelVertexes_cff import *

# legacy pixel vertex reconsruction using the divisive vertex finder
pixelVerticesTask = cms.Task(
    pixelVertices
)

# "Patatrack" pixel ntuplets, fishbone cleaning, Broken Line fit, and density-based vertex reconstruction
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

# HIon modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

# build the pixel vertices in SoA format on the CPU
from RecoTracker.PixelVertexFinding.pixelVertexProducerCUDAPhase1_cfi import pixelVertexProducerCUDAPhase1 as _pixelVerticesCUDA
from RecoTracker.PixelVertexFinding.pixelVertexProducerCUDAPhase2_cfi import pixelVertexProducerCUDAPhase2 as _pixelVerticesCUDAPhase2
from RecoTracker.PixelVertexFinding.pixelVertexProducerCUDAHIonPhase1_cfi import pixelVertexProducerCUDAHIonPhase1 as _pixelVerticesCUDAHIonPhase1

pixelVerticesSoA = SwitchProducerCUDA(
    cpu = _pixelVerticesCUDA.clone(
        pixelTrackSrc = "pixelTracksSoA",
        onGPU = False
    )
)

phase2_tracker.toModify(pixelVerticesSoA,cpu = _pixelVerticesCUDAPhase2.clone(
    pixelTrackSrc = "pixelTracksSoA",
    onGPU = False,
    PtMin = 2.0
))

pp_on_AA.toModify(pixelVerticesSoA,cpu = _pixelVerticesCUDAHIonPhase1.clone(
    pixelTrackSrc = "pixelTracksSoA",
    doSplitting = False,
    onGPU = False,
))

# convert the pixel vertices from SoA to legacy format
from RecoTracker.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA

(pixelNtupletFit).toReplaceWith(pixelVertices, _pixelVertexFromSoA.clone(
    src = "pixelVerticesSoA"
))

(pixelNtupletFit).toReplaceWith(pixelVerticesTask, cms.Task(
    # build the pixel vertices in SoA format on the CPU
    pixelVerticesSoA,
    # convert the pixel vertices from SoA to legacy format
    pixelVertices
))


# "Patatrack" sequence running on the GPU
from Configuration.ProcessModifiers.gpu_cff import gpu

# build pixel vertices in SoA format on the GPU
pixelVerticesCUDA = _pixelVerticesCUDA.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    onGPU = True
)

phase2_tracker.toReplaceWith(pixelVerticesCUDA,_pixelVerticesCUDAPhase2.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    onGPU = True,
    PtMin = 2.0
))

pp_on_AA.toReplaceWith(pixelVerticesCUDA,_pixelVerticesCUDAHIonPhase1.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    doSplitting = False,
    onGPU = True
))

# transfer the pixel vertices in SoA format to the CPU
from RecoTracker.PixelVertexFinding.pixelVerticesSoA_cfi import pixelVerticesSoA as _pixelVerticesSoA
gpu.toModify(pixelVerticesSoA,
    cuda = _pixelVerticesSoA.clone(
        src = cms.InputTag("pixelVerticesCUDA")
    )
)

## GPU vs CPU validation
# force CPU vertexing to use track SoA from CPU chain and not the converted one from GPU chain
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
(pixelNtupletFit & gpu & gpuValidationPixel).toModify(pixelVerticesSoA.cpu,
    pixelTrackSrc = "pixelTracksSoA@cpu"
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
