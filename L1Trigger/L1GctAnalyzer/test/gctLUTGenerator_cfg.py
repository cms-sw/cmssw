from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import sys
import os

# arguments
if (len(sys.argv)>1) :
    key=str(sys.argv[2])
else :
    key='Default'

print("Generating LUT files for GCT key %s" % (key))

if (not ("TNS_ADMIN" in os.environ.keys())):
    print("Please set TNS_ADMIN using :")
    print("export TNS_ADMIN=/nfshome0/popcondev/conddb")


# CMSSW config
process = cms.Process("GctLUTGen")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('l1GctConfigDump')

# Generate dummy L1TriggerKeyList
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


#process.load("L1TriggerConfig.GctConfigProducers.L1GctChannelMaskOnline_cfi")


from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# LUT printer
process.load("L1Trigger.GlobalCaloTrigger.l1GctPrintLuts_cfi")
process.l1GctPrintLuts.jetRanksFilename = cms.untracked.string("GctJetLUT_"+key+".txt")
process.l1GctPrintLuts.hfSumLutFilename = cms.untracked.string("GctHFSumLUT_"+key+".txt")
process.l1GctPrintLuts.htMissLutFilename = cms.untracked.string("GctHtMissLUT_"+key+".txt")


process.p = cms.Path(
    process.l1GctPrintLuts
)



