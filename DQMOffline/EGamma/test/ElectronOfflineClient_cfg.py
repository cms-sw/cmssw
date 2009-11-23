
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

process.load("DQMOffline.EGamma.electronOfflineClient_cfi")
process.dqmElectronOfflineClient.FinalStep = cms.string("AtJobEnd")
process.dqmElectronOfflineClient.InputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.dqmElectronOfflineClient.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])

process.p = cms.Path(process.dqmElectronOfflineClient*process.dqmStoreStats)


