import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test streamer input')

parser.add_argument("--alt", help="Have filter succeed", action="store_true")
parser.add_argument("--ext", help="Switch the order of dependencies", action="store_true")

args = parser.parse_args()


process = cms.Process("TRANSFER")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("WatcherSource",
                            inputDir = cms.string("."),
                            filePatterns = cms.vstring("inDir/.*\.dat"),
                            inprocessDir = cms.string("process"),
                            processedDir = cms.string("processed"),
                            corruptedDir = cms.string("corrupt"),
                            tokenFile = cms.untracked.string("watcherSourceToken"),
                            timeOutInSec = cms.int32(10),
                            verbosity = cms.untracked.int32(1)
)

#process.finishProcessFile = cms.EDAnalyzer("ecallasersortingtest::CreateFileAfterStartAnalyzer",
#                              fileName = cms.untracked.string("watcherSourceToken")
#)

process.a1 = cms.EDAnalyzer("StreamThingAnalyzer",
    product_to_get = cms.string('m1')
)

ids = [cms.EventID(1,0,0), cms.EventID(1,1,0)]
for e in range(10123456789, 10123456839):
    ids.append(cms.EventID(1,1,e))
if args.alt:
    for e in range(15123456789, 15123456839):
        ids.append(cms.EventID(1,1,e))

if args.ext:
    ids.append(cms.EventID(1,1,0))
    ids.append(cms.EventID(1,0,0))
    ids.append(cms.EventID(1,0,0))
    ids.append(cms.EventID(1,1,0))
    for e in range(20123456789, 20123456839):
        ids.append(cms.EventID(1,1,e))

ids.append(cms.EventID(1,1,0))
ids.append(cms.EventID(1,0,0))

process.check = cms.EDAnalyzer("RunLumiEventChecker",
                               eventSequence = cms.untracked.VEventID(ids)
)


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myout.root')
)

process.end = cms.EndPath(process.a1*process.out*process.check)
