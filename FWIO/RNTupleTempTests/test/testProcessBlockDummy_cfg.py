import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMMY")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testProcessBlock5.root'
    )
)

process.dummy = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.p = cms.Path(process.dummy)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testProcessBlockDummy.root'),
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_*_*_DUMMY")
)
process.e = cms.EndPath(process.out)
