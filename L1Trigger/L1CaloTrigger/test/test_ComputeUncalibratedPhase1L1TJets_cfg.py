import FWCore.ParameterSet.Config as cms
from math import pi

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/TTBar_PU200/inputs104X_TTbar_PU200_job1.root",
  )
)

process.load('L1Trigger.L1CaloTrigger.Phase1L1TJetProducer_cfi')

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('myOutputFile.root'),
  outputCommands = cms.untracked.vstring(
    "drop *",
    "keep *_Phase1L1TJetProducer_*_*",
    "keep *_ak4GenJetsNoNu_*_*",
  ),
)

process.p = cms.Path(
  process.Phase1L1TJetProducer
)

process.e = cms.EndPath(process.out)