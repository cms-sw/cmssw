import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(0)
)

from HeterogeneousCore.Producer.testHeterogeneousEDProducerGPU_cfi import testHeterogeneousEDProducerGPU as prod

#process.Tracer = cms.Service("Tracer")
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.prod1 = prod.clone()
process.prod2 = prod.clone(src = "prod1")
process.prod3 = prod.clone(src = "prod1")
process.prod4 = prod.clone()
process.ana = cms.EDAnalyzer("TestHeterogeneousEDProducerAnalyzer",
    src = cms.VInputTag("prod2", "prod3", "prod4")
)

process.t = cms.Task(process.prod1, process.prod2, process.prod3, process.prod4)
process.p = cms.Path(process.ana)
process.p.associate(process.t)

# Example of disabling CUDA device type for one module via configuration
#process.prod4.heterogeneousEnabled_.GPUCuda = False

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
