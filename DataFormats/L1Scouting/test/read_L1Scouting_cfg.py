import FWCore.ParameterSet.Config as cms

import argparse
import sys
import os

parser = argparse.ArgumentParser(prog=sys.argv[0],
    description='Test reading EDM products with L1Scouting data formats',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-n', '--maxEvents', type=int, default=1,
    help='Value of process.maxEvents.input')

parser.add_argument("-i", "--inputFileName", type=str, default="testL1Scouting.root",
    help="Input file name")

parser.add_argument("-o", "--outputFileName", type=str, default="testL1Scouting2.root",
    help="Output file name")

parser.add_argument("--bmtfStubVersion", type=int, default=3,
    help="L1ScoutingBMTFStub data format version")

parser.add_argument("--caloTowerVersion", type=int, default=3,
    help="L1ScoutingCaloTower data format version")

parser.add_argument("--fastJetVersion", type=int, default=3,
    help="L1ScoutingFastJet data format version")

args = parser.parse_args()

if os.path.abspath(args.inputFileName) == os.path.abspath(args.outputFileName):
    raise SystemExit(f">>> Fatal error - the input and output file names point to the same path: {args.inputFileName}")

process = cms.Process("READ")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(f"file:{args.inputFileName}")
)

process.maxEvents.input = args.maxEvents

process.l1ScoutingTestAnalyzer = cms.EDAnalyzer("TestReadL1Scouting",
  bxValues = cms.vuint32(42, 512),
  muonsTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedMuonValues = cms.vint32(1, 2, 3),
  jetsTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedJetValues = cms.vint32(4, 5, 6, 7),
  eGammasTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedEGammaValues = cms.vint32(8, 9, 10),
  tausTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedTauValues = cms.vint32(11, 12),
  bxSumsTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedBxSumsValues = cms.vint32(13),
  bmtfStubClassVersion = cms.int32(args.bmtfStubVersion),
  bmtfStubTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedBmtfStubValues = cms.vint32(1, 2),
  caloTowerClassVersion = cms.int32(args.caloTowerVersion),
  caloTowersTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedCaloTowerValues = cms.vint32(14, 15, 16, 17, 18),
  fastJetClassVersion = cms.int32(args.fastJetVersion),
  fastJetsTag = cms.InputTag("l1ScoutingTestProducer", "", "PROD"),
  expectedFastJetFloatingPointValues = cms.vdouble(19., 20., 21.),
  expectedFastJetIntegralValues = cms.vint32(22, 23, 24)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(args.outputFileName),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.l1ScoutingTestAnalyzer)
process.endPath = cms.EndPath(process.out)
