import FWCore.ParameterSet.Config as cms

silent = True
#silent = False

#includeAnalyzer = True
includeAnalyzer = False

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3
if not silent:
    process.maxEvents.input = 10
    process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
    process.MessageLogger.cerr.INFO.limit = process.MessageLogger.cerr.default.limit


#process.options.numberOfThreads = 4
process.options.numberOfStreams = 0
#process.Tracer = cms.Service("Tracer")

# Flow diagram of the modules
#
#     1   5
#    / \  |
#   2  4  6
#   |
#   3

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

process.load("HeterogeneousCore.CUDATest.prod1Switch_cff")
process.load("HeterogeneousCore.CUDATest.prod5Switch_cff")
process.load("HeterogeneousCore.CUDATest.prod6Switch_cff")

# GPU producers
from HeterogeneousCore.CUDATest.testCUDAProducerGPUFirst_cfi import testCUDAProducerGPUFirst
from HeterogeneousCore.CUDATest.testCUDAProducerGPU_cfi import testCUDAProducerGPU
from HeterogeneousCore.CUDATest.testCUDAProducerGPUEW_cfi import testCUDAProducerGPUEW
from HeterogeneousCore.CUDATest.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU

process.prod2CUDA = testCUDAProducerGPU.clone(src = "prod1CUDA")
process.prod3CUDA = testCUDAProducerGPU.clone(src = "prod2CUDA")
process.prod4CUDA = testCUDAProducerGPUEW.clone(src = "prod1CUDA")

# CPU producers, switched with modules to copy data from GPU to CPU
# (as "on demand" as any other EDProducer, i.e. according to
# consumes() and prefetching). If a separate conversion step is needed
# to get the same data formats as the CPU modules, those are then ones
# that should be replaced-with here.
from HeterogeneousCore.CUDATest.testCUDAProducerCPU_cfi import testCUDAProducerCPU
process.prod2 = SwitchProducerCUDA(
    cpu = testCUDAProducerCPU.clone(src = "prod1"),
    cuda = testCUDAProducerGPUtoCPU.clone(src = "prod2CUDA")
)
process.prod3 = SwitchProducerCUDA(
    cpu = testCUDAProducerCPU.clone(src = "prod2"),
    cuda = testCUDAProducerGPUtoCPU.clone(src = "prod3CUDA")
)
process.prod4 = SwitchProducerCUDA(
    cpu = testCUDAProducerCPU.clone(src = "prod1"),
    cuda = testCUDAProducerGPUtoCPU.clone(src = "prod4CUDA")
)

# GPU analyzer (optionally)
from HeterogeneousCore.CUDATest.testCUDAAnalyzerGPU_cfi import testCUDAAnalyzerGPU
process.anaCUDA = testCUDAAnalyzerGPU.clone(src="prod6CUDA")
if silent:
    process.anaCUDA.minValue = 2.3e7
    process.anaCUDA.maxValue = 2.5e7

process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_prod3_*_*",
        "keep *_prod4_*_*",
        "keep *_prod6_*_*",
    ),
    verbosity = cms.untracked.uint32(0),
)

process.prod2Task = cms.Task(process.prod2, process.prod2CUDA)
process.prod3Task = cms.Task(process.prod3, process.prod3CUDA)
process.prod4Task = cms.Task(process.prod4, process.prod4CUDA)

process.t = cms.Task(
    process.prod1Task,
    process.prod2Task,
    process.prod3Task,
    process.prod4Task,
    process.prod5Task,
    process.prod6Task
)
process.p = cms.Path()
if includeAnalyzer:
    process.p += process.anaCUDA
process.p.associate(process.t)
process.ep = cms.EndPath(process.out)
