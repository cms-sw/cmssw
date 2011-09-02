import FWCore.ParameterSet.Config as cms

process = cms.Process("THIRD")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testEventHistory_2.root')
)

process.intdeque = cms.EDProducer("IntDequeProducer",
    count = cms.int32(12),
    ivalue = cms.int32(21)
)

process.intlist = cms.EDProducer("IntListProducer",
    count = cms.int32(4),
    ivalue = cms.int32(3)
)

process.intset = cms.EDProducer("IntSetProducer",
    start = cms.int32(100),
    stop = cms.int32(110)
)

process.intvec = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(11)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_3.root')
)

process.outother = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_other.root')
)

process.p3 = cms.Path(process.intdeque+process.intlist+process.intset)
process.ep31 = cms.EndPath(process.out)
process.ep32 = cms.EndPath(process.intvec*process.intset*process.outother*process.out*process.outother)
process.epother = cms.EndPath(process.outother)
