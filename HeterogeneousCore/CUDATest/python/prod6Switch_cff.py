import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod6CPU_cfi import prod6CPU as _prod6CPU
from HeterogeneousCore.CUDATest.prod6CUDA_cfi import prod6CUDA
from HeterogeneousCore.CUDATest.prod6FromCUDA_cfi import prod6FromCUDA as _prod6FromCUDA

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

prod6 = SwitchProducerCUDA(
    cpu = _prod6CPU.clone(),
    cuda = _prod6FromCUDA.clone()
)

prod6Task = cms.Task(
    prod6CUDA,
    prod6
)

