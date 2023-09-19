#
# test-o2o-emulator.py
#
# Run O2O for a given configuration key, then run GCT emulator and DQM
#
# NB - masks not included for now!

import FWCore.ParameterSet.Config as cms

import sys
import os
from subprocess import *

# arguments
key=str(sys.argv[1])


# CMSSW process proper
process = cms.Process('testGCTO2OEmulator')


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(1000)
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('l1GctConfigDump')

# Configuration
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# get 
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

# xxxKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
process.L1TriggerKeyDummy.gctKey = cms.string(key)

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.GctConfigProducers.L1GctTSCObjectKeysOnline_cfi")
process.L1GctTSCObjectKeysOnline.subsystemLabel = cms.string('')

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.GctConfigProducers.L1GctJetFinderParamsOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1JetEtScaleOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1HfRingEtScaleOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1HtMissScaleOnline_cfi")

# dummy producer for masks
process.load("L1TriggerConfig.GctConfigProducers.L1GctConfig_cff")
# but this produces masks *and* parameters, so set es_prefer for online params
process.es_prefer = cms.ESPrefer('L1GctJetFinderParamsOnlineProd', 'L1GctJetFinderParamsOnline')

# dump config
process.load('L1TriggerConfig/GctConfigProducers.l1GctConfigDump_cfi')


# GCT Unpacker
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')

# GCT emulator
process.load('L1Trigger.HardwareValidation.L1HardwareValidation_cff')
process.l1compare.COMPARE_COLLS = cms.untracked.vuint32(
# ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
)

# GCT DQM
process.load('DQMServices.Core.DQM_cfg')
process.load('DQM.L1TMonitor.L1TGCT_cfi')
process.l1tgct.disableROOToutput = cms.untracked.bool(False)
process.l1tgct.gctCentralJetsSource = cms.InputTag("gctDigis","cenJets")
process.l1tgct.gctNonIsoEmSource = cms.InputTag("gctDigis","nonIsoEm")
process.l1tgct.gctForwardJetsSource = cms.InputTag("gctDigis","forJets")
process.l1tgct.gctIsoEmSource = cms.InputTag("gctDigis","isoEm")
process.l1tgct.gctEnergySumsSource = cms.InputTag("gctDigis","")
process.l1tgct.gctTauJetsSource = cms.InputTag("gctDigis","tauJets")

# RCT DQM
process.load('DQM.L1TMonitor.L1TRCT_cfi')
process.l1trct.disableROOToutput = cms.untracked.bool(False)
process.l1trct.rctSource = cms.InputTag("gctDigis","")


# GCT EXPERT EMU DQM
process.load('DQM.L1TMonitor.L1TdeGCT_cfi')
process.l1demongct.VerboseFlag = cms.untracked.int32(0)
process.l1demongct.DataEmulCompareSource = cms.InputTag("l1compare")
process.l1demongct.disableROOToutput = cms.untracked.bool( False )
process.l1demongct.HistFile = cms.untracked.string('test-o2o-emulator.root')


process.path = cms.Path (
process.gctDigis +
process.valGctDigis + 
process.l1compare +
process.l1trct + 
process.l1tgct + 
process.l1demongct
#process.l1GctConfigDump
)



# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 1000 ) )

# input files
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring('file:/lookarea_SM/PrivMinidaq.00124879.0001.A.storageManager.00.0000.dat'
#        )
#)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:~/cmssw/Calo_CRAFT09-GR09_31X_V5P_StoppedHSCP-332_v4_RAW-RECO_111039_test.root'
        )
)

