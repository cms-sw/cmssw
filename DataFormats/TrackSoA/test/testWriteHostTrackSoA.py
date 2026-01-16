import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 5

process.trackSoA = cms.EDProducer("TestWriteHostTrackSoA",
    trackSize = cms.uint32(2708)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(sys.argv[1])
)

process.path = cms.Path(process.trackSoA)
process.endPath = cms.EndPath(process.out)
