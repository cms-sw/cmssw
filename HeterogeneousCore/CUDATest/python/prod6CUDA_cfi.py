import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerGPU_cfi import testCUDAProducerGPU as _testCUDAProducerGPU
prod6CUDA = _testCUDAProducerGPU.clone(src = "prod5CUDA")
