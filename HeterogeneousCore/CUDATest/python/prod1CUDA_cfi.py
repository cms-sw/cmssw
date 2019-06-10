import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerGPUFirst_cfi import testCUDAProducerGPUFirst as _testCUDAProducerGPUFirst
prod1CUDA = _testCUDAProducerGPUFirst.clone()
