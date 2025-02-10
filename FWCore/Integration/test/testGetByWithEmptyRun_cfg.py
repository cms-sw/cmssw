# The purpose of this is to test the case where a
# file ends with an empty run (no lumis) and we
# go into another file and process things.
# This tests a rarely hit code path in processRuns.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD3")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [
        'file:testGetByRunsMode.root',
        'file:testGetBy1.root'
    ],
    inputCommands = [
        'keep *',
        'drop *_*_*_PROD2'
    ]
)

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(fileName = 'testGetByWithEmptyRun.root')

from FWCore.Framework.modules import RunLumiEventAnalyzer
process.test = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
1, 0, 0,
1, 0, 0,
1, 0, 0,
1, 1, 0,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 0,
1, 0, 0
]
)

process.p1 = cms.Path(process.test)

process.e1 = cms.EndPath(process.out)
