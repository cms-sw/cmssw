
import sys
import os
import dbs_discovery
import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source ("PoolSource",
  fileNames = cms.untracked.vstring(),
  secondaryFileNames = cms.untracked.vstring()
)

process.source.fileNames.extend(dbs_discovery.search())

from RecoEgamma.Examples.fakeAnalyzerStdBiningParameters_cff import *
from RecoEgamma.Examples.fakeAnalyzerFineBiningParameters_cff import *

process.gsfElectronFakeAnalysis = cms.EDAnalyzer("GsfElectronMCFakeAnalyzer",
  beamSpot = cms.InputTag('offlineBeamSpot'),
  electronCollection = cms.InputTag("gsfElectrons"),
  matchingObjectCollection = cms.InputTag("iterativeCone5GenJets"),
  readAOD = cms.bool(False),
  outputFile = cms.string(os.environ['TEST_OUTPUT_FILE']),
  MaxPt = cms.double(100.0),
  DeltaR = cms.double(0.3),
  MaxAbsEta = cms.double(2.5),
  HistosConfigurationFake = cms.PSet(
  fakeAnalyzerStdBiningParameters
  #fakeAnalyzerFineBiningParameters
  )
)

process.p = cms.Path(process.gsfElectronFakeAnalysis)

