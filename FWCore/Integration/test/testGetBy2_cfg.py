import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

process.Tracer = cms.Service('Tracer')

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
