import FWCore.ParameterSet.Config as cms
from L1Trigger.L1CaloTrigger.caloEtaSegmentation import caloEtaSegmentation
from math import pi
from copy import deepcopy

import pickle as pkl

import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

process = cms.Process("UncalibratedL1TJetPhase1Producer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:/hdfs/user/sb17498/CMS_Phase_2/jetMETStudies/GeneratePfClustersAndCandidatesFromQCD_PU200/GeneratePfClustersAndCandidatesFromQCD_PU200_3720957.106.root",
  )
)

process.Phase1L1TJetFromPfClustersProducer = cms.EDProducer('L1TJetPhase1Producer',
  inputCollectionTag = ak4JetFromPfClustersParameters.src,
  etaBinning = caloEtaSegmentation,
  nBinsPhi = cms.uint32(72),
  phiLow = cms.double(-pi),
  phiUp = cms.double(pi),
  jetIEtaSize = cms.uint32(5),
  jetIPhiSize = cms.uint32(5),
  seedPtThreshold = cms.double(4), # GeV
  puSubtraction = cms.bool(False),
  outputCollectionName = cms.string("Phase1L1TJetFromPfClusters"),
  vetoZeroPt = cms.bool(True)
)

process.p = cms.Path(
  process.Phase1L1TJetFromPfClustersProducer
)

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('myOutputFile.root'),
  outputCommands = cms.untracked.vstring(
    "keep *",
  ),
)

process.e = cms.EndPath(process.out)
