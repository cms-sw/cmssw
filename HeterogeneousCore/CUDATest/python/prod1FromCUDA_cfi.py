import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU as _testCUDAProducerGPUtoCPU
prod1FromCUDA = _testCUDAProducerGPUtoCPU.clone(src = "prod1CUDA")
