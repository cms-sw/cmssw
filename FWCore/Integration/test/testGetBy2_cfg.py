import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

process.Tracer = cms.Service('Tracer',
                             dumpContextForLabels = cms.untracked.vstring('intProducer'),
                             dumpNonModuleContext = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        Tracer = cms.untracked.PSet(
            limit = cms.untracked.int32(100000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testGetBy1.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGetBy2.root')
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(2))

process.intProducerU = cms.EDProducer("IntProducer", ivalue = cms.int32(20))

process.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(21)
)

process.t = cms.Task(process.intProducerU, process.intVectorProducer)

process.p = cms.Path(process.intProducer, process.t)

process.e = cms.EndPath(process.out)
