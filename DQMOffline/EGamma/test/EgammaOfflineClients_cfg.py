
import sys
import os
import electronDbsDiscovery
import FWCore.ParameterSet.Config as cms

process = cms.Process("testEgammaOfflineClients")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")

process.load("DQMOffline.EGamma.egammaPostProcessing_cff")

process.photonOfflineClient.batch = cms.bool(True)
process.photonOfflineClient.InputFileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'])

process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
process.dqmSaver.workflow = '/'+os.environ['TEST_DATASET']+'/'+os.environ['DBS_RELEASE']+'-'+os.environ['DBS_COND']+'/DQMOFFLINE'
process.dqmsave_step = cms.Path(process.DQMSaver)

process.p = cms.Path(process.egammaPostprocessing*process.dqmStoreStats*process.DQMSaver)


