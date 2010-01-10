
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

process.load("DQMOffline.EGamma.electronOfflineClientSequence_cff")
process.dqmElectronClientAllElectrons.InputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.dqmElectronClientAllElectrons.FinalStep = cms.string("AtJobEnd")
process.dqmElectronClientSelectionEt.FinalStep = cms.string("AtJobEnd")
process.dqmElectronClientSelectionEtIso.FinalStep = cms.string("AtJobEnd")
process.dqmElectronClientSelectionEtIso.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.dqmElectronClientSelectionEtIsoElID.FinalStep = cms.string("AtJobEnd")
process.dqmElectronClientTagAndProbe.FinalStep = cms.string("AtJobEnd")

process.p = cms.Path(process.electronOfflineClientSequence*process.dqmStoreStats)


