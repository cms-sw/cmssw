import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test L1 Scouting data formats')

parser.add_argument("--inputFile", type=str, help="Input file name (default: testL1Scouting.root)", default="testL1Scouting.root")
parser.add_argument("--bmtfStubVersion", type=int, help="track data format version (default: 3)", default=3)
args = parser.parse_args()

process = cms.Process("READ")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+args.inputFile))
process.maxEvents.input = 1

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
  expectedBmtfStubValues = cms.vint32(1, 2)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testL1Scouting2.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.l1ScoutingTestAnalyzer)
process.endPath = cms.EndPath(process.out)
