import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod5CPU_cfi import prod5CPU as _prod5CPU
from HeterogeneousCore.CUDATest.prod5CUDA_cfi import prod5CUDA
from HeterogeneousCore.CUDATest.prod5FromCUDA_cfi import prod5FromCUDA as _prod5FromCUDA

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

prod5 = SwitchProducerCUDA(
    cpu = _prod5CPU.clone(),
    cuda = _prod5FromCUDA.clone()
)

prod5Task = cms.Task(
    prod5CUDA,
    prod5
)
