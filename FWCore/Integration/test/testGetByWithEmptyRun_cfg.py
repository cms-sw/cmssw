# The purpose of this is to test the case where a
# file ends with an empty run (no lumis) and we
# go into another file and process things.
# This tests a rarely hit code path in processRuns.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD3")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testGetByRunsMode.root',
        'file:testGetBy1.root'
    ),
    inputCommands=cms.untracked.vstring(
        'keep *',
        'drop *_*_*_PROD2'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGetByWithEmptyRun.root')
)

process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
                              verbose = cms.untracked.bool(True),
                              expectedRunLumiEvents = cms.untracked.vuint32(
1, 0, 0,
1, 0, 0,
1, 0, 0,
1, 1, 0,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 0,
1, 0, 0
)
)

process.p1 = cms.Path(process.test)

process.e1 = cms.EndPath(process.out)
