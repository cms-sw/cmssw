
import sys
import os
import dbs_discovery
import FWCore.ParameterSet.Config as cms

process = cms.Process("testElectronAnalyzer")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
#from DQMServices.Components.DQMStoreStats_cfi import *
#dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dbs_discovery.search())

process.load("DQMOffline.EGamma.electronAnalyzerSequence_cff")
process.dqmElectronAnalysisAllElectrons.FinalStep = cms.string("AtJobEnd")
process.dqmElectronAnalysisSelectionEt.FinalStep = cms.string("AtJobEnd")
process.dqmElectronAnalysisSelectionEtIso.FinalStep = cms.string("AtJobEnd")
process.dqmElectronAnalysisSelectionEtIso.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.dqmElectronAnalysisSelectionEtIsoElID.FinalStep = cms.string("AtJobEnd")
process.dqmElectronAnalysisTagAndProbe.FinalStep = cms.string("AtJobEnd")

process.p = cms.Path(process.electronAnalyzerSequence*process.dqmStoreStats)
