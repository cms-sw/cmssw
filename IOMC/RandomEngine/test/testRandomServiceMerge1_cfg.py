import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRandomService1.root',
        'file:testRandomService2.root',
        'file:testRandomService3.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRandomServiceMerge1.root')
)

process.o = cms.EndPath(process.out)
