
import sys
import os
import electronDbsDiscovery
import FWCore.ParameterSet.Config as cms

process = cms.Process("testEgammaAnalyzers")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
#from DQMServices.Components.DQMStoreStats_cfi import *
#dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(electronDbsDiscovery.search())

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQMOffline.EGamma.egammaDQMOffline_cff")

process.dqmElectronTagProbeAnalysis.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])

process.p = cms.Path(process.egammaDQMOffline*process.dqmStoreStats)
