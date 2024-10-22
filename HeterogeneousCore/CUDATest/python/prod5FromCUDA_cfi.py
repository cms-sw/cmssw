import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU as _testCUDAProducerGPUtoCPU
prod5FromCUDA = _testCUDAProducerGPUtoCPU.clone(src = "prod5CUDA")
