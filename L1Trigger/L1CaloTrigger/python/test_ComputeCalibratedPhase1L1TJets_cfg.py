import FWCore.ParameterSet.Config as cms
from math import pi

process = cms.Process("CalibratedL1TJetPhase1Producer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/ComputeUncalibratedPhase1AndAK4L1TJetsFromPfClustersAndCandidates_QCD_PU200_Puppi_NoZeroPTJets_0.4Square/ComputeUncalibratedPhase1AndAK4L1TJetsFromPfClustersAndCandidates_QCD_PU200_Puppi_FinerGranularity_0.8Square_3751805.0.root",
  )
)

process.load('L1Trigger.L1CaloTrigger.ComputeCalibratedPhase1L1TJets_cfi')

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('myOutputFile.root'),
  outputCommands = cms.untracked.vstring(
    "keep *",
  ),
)

process.e = cms.EndPath(process.out)