import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerCPU_cfi import testCUDAProducerCPU as _testCUDAProducerCPU
prod5CPU = _testCUDAProducerCPU.clone()
