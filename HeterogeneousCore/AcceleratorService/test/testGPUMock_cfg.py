import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(0)
)


#process.Tracer = cms.Service("Tracer")
process.AcceleratorService = cms.Service("AcceleratorService")
process.prod1 = cms.EDProducer('TestAcceleratorServiceProducerGPUMock')
process.prod2= cms.EDProducer('TestAcceleratorServiceProducerGPUMock',
    src = cms.InputTag("prod1")
)

#process.t = cms.Task(process.prod1, process.prod2)

process.eca = cms.EDAnalyzer("EventContentAnalyzer",
    getData = cms.untracked.bool(True),
    getDataForModuleLabels = cms.untracked.vstring("producer"),
    listContent = cms.untracked.bool(True),
)
process.p = cms.Path(process.prod1+process.prod2)#+process.eca)
#process.p.associate(process.t)
