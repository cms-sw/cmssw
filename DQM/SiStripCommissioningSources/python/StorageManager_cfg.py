import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.source = cms.Source("FragmentInput")

process.out = cms.OutputModule("EventStreamFileWriter",
    streamLabel = cms.string('A'),
    max_queue_depth = cms.untracked.int32(5),
    maxSize = cms.int32(1000),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(False),
    max_event_size = cms.untracked.int32(25000000),
    SelectHLTOutput = cms.untracked.string('consumer')
)

process.e1 = cms.EndPath(process.out)

