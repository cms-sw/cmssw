import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod1CPU_cfi import prod1CPU as _prod1CPU
from HeterogeneousCore.CUDATest.prod1CUDA_cfi import prod1CUDA
from HeterogeneousCore.CUDATest.prod1FromCUDA_cfi import prod1FromCUDA as _prod1FromCUDA

from Configuration.ProcessModifiers.gpu_cff import gpu

prod1 = _prod1CPU.clone()
gpu.toReplaceWith(prod1, _prod1FromCUDA)

prod1Task = cms.Task(
    prod1CUDA,
    prod1
)
