import FWCore.ParameterSet.Config as cms

process = cms.Process("SECOND")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testEventHistory_1.root')
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

process.filt55 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process.filt75 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(75)
)

process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('f55')
    ),
    fileName = cms.untracked.string('testEventHistory_2.root')
)

process.s = cms.Sequence(process.intdeque+process.intlist+process.intset+process.intvec)
process.f55 = cms.Path(process.s*process.filt55)
process.f75 = cms.Path(process.s*process.filt75)
process.ep2 = cms.EndPath(process.out)

process.sched = cms.Schedule(process.f55, process.f75, process.ep2)
