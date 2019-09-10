import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerGPUEWTask_cfi import testCUDAProducerGPUEWTask as _testCUDAProducerGPUEWTask
prod6CUDA = _testCUDAProducerGPUEWTask.clone(src = "prod5CUDA")
