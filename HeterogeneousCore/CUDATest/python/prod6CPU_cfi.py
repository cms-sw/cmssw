import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerCPU_cfi import testCUDAProducerCPU as _testCUDAProducerCPU
prod6CPU = _testCUDAProducerCPU.clone(src = "prod5")
