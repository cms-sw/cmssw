import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description='Test reduced ProcessHistory')
parser.add_argument("--input", type=str, help="Input file")
parser.add_argument("--bypassVersionCheck", action="store_true", help="Bypass version check")
parser.add_argument("--expectNewLumi", action="store_true", help="Set this if a new lumi is expected between the original files")
parser.add_argument("--expectNewRun", action="store_true", help="Set this if a new run is expected between the original files")

args = parser.parse_args()

process = cms.Process("READ")

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [f"file:{args.input}"],
    bypassVersionCheck = args.bypassVersionCheck,
)

from FWCore.Framework.modules import TestMergeResults, RunLumiEventAnalyzer
process.testmerge = TestMergeResults(
    expectedBeginRunProd = [10001, 20004, 10003],
    expectedEndRunProd = [100001, 200004, 100003],
    expectedBeginLumiProd = [101, 204, 103],
    expectedEndLumiProd = [1001, 2004, 1003],

    expectedBeginRunNew = [10001, 10002, 10003],
    expectedEndRunNew = [100001, 100002, 100003],
    expectedBeginLumiNew = [101, 102, 103],
    expectedEndLumiNew = [1001, 1002, 1003],
    expectedProcessHistoryInRuns = [
        'PROD',
        'MERGETWOFILES',
        'READ'
    ]
)
def setWithMergeAndCopyEntry(p, value):
    p[1] = value
    p[3:5] = p[0:2]
if args.expectNewLumi or args.expectNewRun:
    setWithMergeAndCopyEntry(process.testmerge.expectedBeginRunProd, 10002)
    setWithMergeAndCopyEntry(process.testmerge.expectedEndRunProd, 100002)
    setWithMergeAndCopyEntry(process.testmerge.expectedBeginLumiProd, 102)
    setWithMergeAndCopyEntry(process.testmerge.expectedEndLumiProd, 1002)

process.test = RunLumiEventAnalyzer(
    expectedRunLumiEvents = [
        1, 0, 0, # beginRun
        1, 1, 0, # beginLumi
        1, 1, 1,
        1, 1, 2,
        1, 1, 3,
        1, 1, 4,
        1, 1, 5,
        1, 1, 6,
        1, 1, 7,
        1, 1, 8,
        1, 1, 9,
        1, 1, 10,
        1, 1, 101,
        1, 1, 102,
        1, 1, 103,
        1, 1, 104,
        1, 1, 105,
        1, 1, 106,
        1, 1, 107,
        1, 1, 108,
        1, 1, 109,
        1, 1, 110,
        1, 1, 0, # endLumi
        1, 0, 0, # endRun
    ]
)
endFirstFileIndex = 3*(10+2)
if args.expectNewLumi:
    process.test.expectedRunLumiEvents = process.test.expectedRunLumiEvents[:endFirstFileIndex] + [
        1, 1, 0, # endLumi
        1, 0, 0, # endRun
        1, 0, 0, # beginRun
        1, 1, 0, # beginLumi
        1, 1, 201,
        1, 1, 202,
        1, 1, 203,
        1, 1, 204,
        1, 1, 205,
        1, 1, 206,
        1, 1, 207,
        1, 1, 208,
        1, 1, 209,
        1, 1, 210,
        1, 1, 0, # endLumi
        1, 0, 0, # endRun
    ]
elif args.expectNewRun:
    process.test.expectedRunLumiEvents = process.test.expectedRunLumiEvents[:endFirstFileIndex] + [
        1, 1, 0, # endLumi
        1, 0, 0, # endRun
        1, 0, 0, # beginRun
        1, 2, 0, # beginLumi
        1, 2, 201,
        1, 2, 202,
        1, 2, 203,
        1, 2, 204,
        1, 2, 205,
        1, 2, 206,
        1, 2, 207,
        1, 2, 208,
        1, 2, 209,
        1, 2, 210,
        1, 2, 0, # endLumi
        1, 0, 0, # endRun
    ]

process.p = cms.Path(
    process.testmerge +
    process.test
)
