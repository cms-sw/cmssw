from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# LUT generator process
process = cms.Process("LUTgen")

# just run once
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

# setup message logging to cout, without any filtering
#process.MessageLogger = cms.Service("MessageLogger",
#     destinations = cms.untracked.vstring('cout'),
#     cout = cms.untracked.PSet(
#         threshold = cms.untracked.string('DEBUG'), ## DEBUG mode 
#
#         DEBUG = cms.untracked.PSet( 
#             limit = cms.untracked.int32(-1)          ## DEBUG mode, all messages  
#             #limit = cms.untracked.int32(10)         ## DEBUG mode, max 10 messages 
#         ),
#         INFO = cms.untracked.PSet(
#             limit = cms.untracked.int32(-1)
#         )
#     ),
#     debugModules = cms.untracked.vstring('*'), ## DEBUG mode 
#)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

##############################
# CondTools key lookup magic #
##############################

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# # from https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1ConfigOnlineProd

# Alternative 1 :
# For known object keys:

# print "Specifying parameters directly"
# process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
# process.L1TriggerKeyDummy.objectKeys = cms.VPSet(
#     cms.PSet(
#         record = cms.string('L1MuGMTParametersRcd'),
#         type = cms.string('L1MuGMTParameters'),
#         key = cms.string('default1')),
#     cms.PSet(
#         record = cms.string('L1MuTriggerScalesRcd'),
#         type = cms.string('L1MuTriggerScales'),
#         key = cms.string('...')),
#         cms.PSet(
#         record = cms.string('L1MuTriggerPtScaleRcd'),
#         type = cms.string('L1MuTriggerPtScale'),
#         key = cms.string('...'))    
# )



# Alternative 2:
# For a known subsystem key (MySubsystemKey):
# [x] Works!

print("Specifying GMT key")
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

# xxxKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
process.L1TriggerKeyDummy.gmtKey = cms.string('test20080429')

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersKeysOnlineProd_cfi")
process.L1MuGMTParametersKeysOnlineProd.subsystemLabel = cms.string('softwareConfig')
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScaleKeysOnlineProd_cfi")
process.L1MuTriggerScaleKeysOnlineProd.subsystemLabel = cms.string('scales')

process.load("CondTools.L1Trigger.L1TriggerKeyOnline_cfi")
process.L1TriggerKeyOnline.subsystemLabels = cms.vstring('softwareConfig', 'scales')

# # Alternative 3:
# # For a known TSC key (MyTSCKey):
# # [x] Works!

# print "Specifying TSC key"
# process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
# process.L1SubsystemKeysOnline.tscKey = cms.string( 'TSC_CRUZET2_080613_GTmuon_GMTDTRPC5CSC5_CSCclosedwindow_DTTFtopbot_RPC_LUM_GCT_RCTH' )
 
# # Subclass of L1ObjectKeysOnlineProdBase.
# process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersKeysOnlineProd_cfi")
# process.L1MuGMTParametersKeysOnlineProd.subsystemLabel = cms.string('softwareConfig')
# process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScaleKeysOnlineProd_cfi")
# process.L1MuTriggerScaleKeysOnlineProd.subsystemLabel = cms.string('scales')

# process.load("CondTools.L1Trigger.L1TriggerKeyOnline_cfi")
# process.L1TriggerKeyOnline.subsystemLabels = cms.vstring('softwareConfig', 'scales')

######################
# GMT emulator setup #
######################

# load external parameter data (TODO: Get this from DB as well)
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi")
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff")

# load online producers for scales
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleOnlineProducer_cfi")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")

# load online producer for GMT parameters
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersOnlineProducer_cfi")

# ignore CMSSW version mismatches!
process.L1MuGMTParametersOnlineProducer.ignoreVersionMismatch = True

# load the GMT simulator 
print("Before load")
process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cfi")
print("After load")

# Clear event data
process.gmtDigis.DTCandidates = cms.InputTag("none", "")
process.gmtDigis.CSCCandidates = cms.InputTag("none", "")
process.gmtDigis.RPCbCandidates = cms.InputTag("none", "")
process.gmtDigis.RPCfCandidates = cms.InputTag("none", "")

# GMT emulator debugging
process.gmtDigis.Debug = cms.untracked.int32(1)

# Tell the emulator to generate LUTs
process.gmtDigis.WriteLUTsAndRegs = cms.untracked.bool(True)

# And run!
process.path = cms.Path(process.gmtDigis)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)



