
import sys
import os
import dbs_discovery
import FWCore.ParameterSet.Config as cms

process = cms.Process("testElectronOfflineClient")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")

process.load("DQMOffline.EGamma.electronClientSequence_cff")
process.dqmElectronOfflineClient0.InputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.dqmElectronOfflineClient0.FinalStep = cms.string("AtJobEnd")
process.dqmElectronOfflineClient1.FinalStep = cms.string("AtJobEnd")
process.dqmElectronOfflineClient2.FinalStep = cms.string("AtJobEnd")
process.dqmElectronOfflineClient3.FinalStep = cms.string("AtJobEnd")
process.dqmElectronOfflineClient4.FinalStep = cms.string("AtJobEnd")
process.dqmElectronOfflineClient4.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])

process.p = cms.Path(process.electronClientSequence*process.dqmStoreStats)


