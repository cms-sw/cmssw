import FWCore.ParameterSet.Config as cms

process = cms.Process("READTESTMERGE")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

from IOPool.TFileAdaptor.modules import AdaptorConfig
process.add_(AdaptorConfig(stats = False))

process.maxEvents.input = -1

from IOPool.Output.modules import PoolOutputModule
process.output = PoolOutputModule(fileName = 'ReadFastMerge_out.root')

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [
      'file:FastMerge_out.root',
      'file:FastMergeRL_out.root',
      'file:FastMergeR_out.root',
    ]
)

from FWCore.Framework.modules import RunLumiEventAnalyzer
process.test = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
100, 0, 0,
100, 1, 0,
100, 1, 1,
100, 1, 2,
100, 1, 3,
100, 1, 4,
100, 1, 5,
100, 1, 6,
100, 1, 7,
100, 1, 8,
100, 1, 9,
100, 1, 10,
100, 1, 0,
100, 0, 0,
200, 0, 0,
200, 1, 0,
200, 1, 100,
200, 1, 101,
200, 1, 102,
200, 1, 103,
200, 1, 104,
200, 1, 105,
200, 1, 106,
200, 1, 107,
200, 1, 108,
200, 1, 109,
200, 1, 110,
200, 1, 111,
200, 1, 112,
200, 1, 113,
200, 1, 114,
200, 1, 0,
200, 0, 0,
100, 0, 0,
100, 1, 0,
100, 1, 0,
100, 0, 0,
200, 0, 0,
200, 1, 0,
200, 1, 0,
200, 0, 0,
100, 0, 0,
100, 0, 0,
200, 0, 0,
200, 0, 0
]
)

process.path1 = cms.Path(process.test)

process.ep = cms.EndPath(process.output)
