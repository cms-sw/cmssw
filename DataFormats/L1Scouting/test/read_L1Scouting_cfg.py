import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))
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
  expectedBxSumsValues = cms.vint32(13)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testL1Scouting2.root')
)

process.path = cms.Path(process.l1ScoutingTestAnalyzer)
process.endPath = cms.EndPath(process.out)