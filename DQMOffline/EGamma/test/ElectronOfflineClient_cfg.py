
import sys
import os
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
#process.dqmElectronClientTagAndProbe.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])

process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
process.dqmSaver.workflow = os.environ['TEST_WORKFLOW']
process.dqmsave_step = cms.Path(process.DQMSaver)

process.p = cms.Path(process.electronOfflineClientSequence*process.dqmStoreStats*process.DQMSaver)


