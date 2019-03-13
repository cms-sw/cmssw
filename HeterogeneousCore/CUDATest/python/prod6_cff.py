import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod6CPU_cfi import prod6CPU as _prod6CPU
from HeterogeneousCore.CUDATest.prod6CUDA_cfi import prod6CUDA
from HeterogeneousCore.CUDATest.prod6FromCUDA_cfi import prod6FromCUDA as _prod6FromCUDA

from Configuration.ProcessModifiers.gpu_cff import gpu

prod6 = _prod6CPU.clone()
gpu.toReplaceWith(prod6, _prod6FromCUDA)

prod6Task = cms.Task(
    prod6CUDA,
    prod6
)

