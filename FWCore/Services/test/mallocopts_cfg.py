import FWCore.ParameterSet.Config as cms

process = cms.Process('Sim')

process.InitRootHandlers = cms.Service('InitRootHandlers',
    UnloadRootSigHandler = cms.untracked.bool(True)
)

process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
    M_MMAP_MAX = cms.untracked.int32(100001),
    M_TRIM_THRESHOLD = cms.untracked.int32(100002),
    M_TOP_PAD = cms.untracked.int32(100003),
    M_MMAP_THRESHOLD = cms.untracked.int32(100004),
    dump = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source('EmptySource')

process.thing = cms.EDProducer('ThingProducer')

process.p = cms.Path(process.thing)
