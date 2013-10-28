import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

process.Tracer = cms.Service('Tracer',
                             dumpContextForLabels = cms.untracked.vstring('intProducer'),
                             dumpNonModuleContext = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations   = cms.untracked.vstring('cout',
                                           'cerr'
    ),
    categories = cms.untracked.vstring(
        'Tracer'
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet (
            limit = cms.untracked.int32(0)
        ),
        Tracer = cms.untracked.PSet(
            limit=cms.untracked.int32(100000000)
        )
    )
)

process.options = cms.untracked.PSet( allowUnscheduled = cms.untracked.bool(True) )

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

process.p = cms.Path(process.intProducer)

process.e = cms.EndPath(process.out)
