import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READMERGE")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.AdaptorConfig = cms.Service("AdaptorConfig",
    stats = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
      'file:ReadMerge_out.root'
    )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:"+sys.argv[1])
)

process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
100,   0,   0,
100, 100,   0,
100, 100, 100,
100, 100,   0,
100,   0,   0,
  1,   0,   0,  
  1,   1,   0,
  1,   1,  11,
  1,   1,  12,
  1,   1,  13,
  1,   1,  14,
  1,   1,  15,
  1,   1,  16,
  1,   1,  17,
  1,   1,  18,
  1,   1,  19,
  1,   1,  20,
  1,   1, 102,
  1,   1,   0,
  1,   0,   0,
  1,   0,   0,
  1,   1,   0,
  1,   1,  21,
  1,   1,  22,
  1,   1,  23,
  1,   1,  24,
  1,   1,  25,
  1,   1,   1,
  1,   1,   2,
  1,   1,   3,
  1,   1,   4,
  1,   1,   5,
  1,   1,   6,
  1,   1,   7,
  1,   1,   8,
  1,   1,   9,
  1,   1,  10,
  1,   1,   0,
  1,   2,   0,
  1,   2,   0,
  1,   0,   0,
  2,   0,   0,
  2,   1,   0,
  2,   1,   1,
  2,   1,   2,
  2,   1,   3,
  2,   1,   4,
  2,   1,   5,
  2,   1,   0,
  2,   0,   0
)
)

process.test.expectedRunLumiEvents.extend([
 11,   0,   0,
 11,   1,   0,
 11,   1,   1,
 11,   1,   0,
 11,   2,   0,
 11,   2,   1,
 11,   2,   2,
 11,   2,   3,
 11,   2,   0,
 11,   3,   0,
 11,   3,   4,
 11,   3,   5,
 11,   3,   6,
 11,   3,   0,
 11,   4,   0,
 11,   4,   7,
 11,   4,   8,
 11,   4,   9,
 11,   4,   0,
 11,   0,   0,
 12,   0,   0,
 12,   2,   0,
 12,   2,   1,
 12,   2,   2,
 12,   2,   3,
 12,   2,   0,
 12,   3,   0,
 12,   3,   4,
 12,   3,   5,
 12,   3,   6,
 12,   3,   0,
 12,   4,   0,
 12,   4,   7,
 12,   4,   8,
 12,   4,   9,
 12,   4,   0,
 12,   0,   0,
 13,   0,   0,
 13,   2,   0,
 13,   2,   1,
 13,   2,   2,
 13,   2,   0,
 13,   0,   0,
1000,  0,   0,
1000,  1,   0,
1000,  1,   0,
1000,  0,   0,
1001,  0,   0,
1001,  1,   0,
1001,  1,   0,
1001,  0,   0,
1002,  0,   0,
1002,  1,   0,
1002,  1,   0,
1002,  0,   0,
# Between ~3_1_0  and 3_7_X these following are not in the input file
# because runs with no lumis in the input were always dropped.
# The test passes, it just never gets past this point.
2000,  0,   0,
2000,  0,   0,
2001,  0,   0,
2001,  0,   0,
2002,  0,   0,
2002,  0,   0
])

process.path1 = cms.Path(process.test)

process.ep = cms.EndPath(process.output)
