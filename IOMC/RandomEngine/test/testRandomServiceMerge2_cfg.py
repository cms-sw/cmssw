import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testMultiProcess_0.root',
        'file:testMultiProcess_1.root',
        'file:testMultiProcess_2.root'
    ),
    skipBadFiles = cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRandomServiceMerge2.root')
)

process.o = cms.EndPath(process.out)
