import FWCore.ParameterSet.Config as cms
from math import pi
from copy import deepcopy

import pickle as pkl

import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

process = cms.Process("CalibratedL1TJetPhase1Producer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/ComputeUncalibratedPhase1AndAK4L1TJetsFromPfClustersAndCandidates_QCD_PU200_Puppi_NoZeroPTJets_0.4Square/ComputeUncalibratedPhase1AndAK4L1TJetsFromPfClustersAndCandidates_QCD_PU200_Puppi_FinerGranularity_0.8Square_3751805.0.root",
  )
)

process.TFileService = cms.Service('TFileService', fileName = cms.string("Histograms.root"))

with open('/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/CalibrationFactors/Calibration_Tree_Calibration_QCD_PU200_Puppi_NoZeroPTJets_0.4Square_SeedThreshold10_MatchAK4GenJetWithPhase1L1TJetFromPfClusters_3753205.pickle', 'rb') as f:
  Calibration_MatchAK4GenJetWithAK4JetFromPfClusters = pkl.load(f)

process.CalibratePhase1L1TJetFromPfClusters = cms.EDProducer('ApplyCalibrationFactors',
  inputCollectionTag = cms.InputTag("Phase1L1TJetFromPfClustersProducer", "Phase1L1TJetFromPfClusters", "UncalibratedL1TJetPhase1Producer"),
  absEtaBinning = cms.vdouble([p.etaMin.value() for p in Calibration_MatchAK4GenJetWithPhase1L1TJetFromPfClusters] + [Calibration_MatchAK4GenJetWithPhase1L1TJetFromPfClusters[-1].etaMax.value()]),
  calibration = Calibration_MatchAK4GenJetWithPhase1L1TJetFromPfClusters,
  outputCollectionName = cms.string("CalibratedPhase1L1TJetFromPfClusters")
)

CalibrateAK4JetFromPfClusters_path = "/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/CalibrationFactors/Calibration_Tree_Calibration_QCD_PU200_Puppi_NoZeroPTJets_0.4Square_MatchAK4GenJetWithAK4JetFromPfClusters_3752474.0.pickle"
CalibrateAK4JetFromPfCandidates_path = "/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/CalibrationFactors/Calibration_Tree_Calibration_QCD_PU200_Puppi_NoZeroPTJets_0.4Square_MatchAK4GenJetWithAK4JetFromPfCandidates_3752475.0.pickle"
CalibratePhase1L1TJetFromPfClusters_path = "/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/CalibrationFactors/Calibration_Tree_Calibration_QCD_PU200_Puppi_NoZeroPTJets_0.4Square_MatchAK4GenJetWithPhase1L1TJetFromPfClusters_3752476.0.pickle"
CalibratePhase1L1TJetFromPfCandidates_path = "/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/CalibrationFactors/Calibration_Tree_Calibration_QCD_PU200_Puppi_NoZeroPTJets_0.4Square_MatchAK4GenJetWithPhase1L1TJetFromPfCandidates_3752477.0.pickle"

with open(CalibratePhase1L1TJetFromPfClusters_path, 'rb') as f:
  process.CalibratePhase1L1TJetFromPfClusters.calibration = pkl.load(f)
with open(CalibratePhase1L1TJetFromPfCandidates_path, 'rb') as f:
  process.CalibratePhase1L1TJetFromPfCandidates.calibration = pkl.load(f)
with open(CalibrateAK4JetFromPfClusters_path, 'rb') as f:
  process.CalibrateAK4JetFromPfClusters.calibration = pkl.load(f)
with open(CalibrateAK4JetFromPfCandidates_path, 'rb') as f:
  process.CalibrateAK4JetFromPfCandidates.calibration = pkl.load(f)

process.p = cms.Path(
  process.CalibratePhase1L1TJetFromPfClusters
)

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('myOutputFile.root'),
  outputCommands = cms.untracked.vstring(
    "keep *",
  ),
)

process.e = cms.EndPath(process.out)
