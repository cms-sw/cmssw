import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 5

process.hitSoA = cms.EDProducer("TestWriteHostHitSoA",
    hitSize = cms.uint32(2708),
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(sys.argv[1]),
)

process.path = cms.Path(process.hitSoA)
process.endPath = cms.EndPath(process.out)

