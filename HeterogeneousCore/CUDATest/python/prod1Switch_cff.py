import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod1CPU_cfi import prod1CPU as _prod1CPU
from HeterogeneousCore.CUDATest.prod1CUDA_cfi import prod1CUDA
from HeterogeneousCore.CUDATest.prod1FromCUDA_cfi import prod1FromCUDA as _prod1FromCUDA

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

prod1 = SwitchProducerCUDA(
    cpu = _prod1CPU.clone(),
    cuda = _prod1FromCUDA.clone()
)

prod1Task = cms.Task(
    prod1CUDA,
    prod1
)
