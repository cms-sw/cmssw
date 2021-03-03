import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleOnlineProducer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.getter1 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1MuTriggerPtScaleRcd'),
   data = cms.vstring('L1MuTriggerPtScale')
   )),
   verbose = cms.untracked.bool(True)
)

process.getter2 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1MuTriggerScalesRcd'),
   data = cms.vstring('L1MuTriggerScales')
   )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter1 + process.getter2)


# # from https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1ConfigOnlineProd

# Alternative 1 :
# For a known object key (MyObjectKey):

# process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
# process.L1TriggerKeyDummy.objectKeys = cms.VPSet(
#     cms.PSet(
#         record = cms.string('L1MuTriggerPtScaleRcd'),
#         type = cms.string('L1MuTriggerPtScale'),
#         key = cms.string('L1T_Scales_20080926_startup')
#         ),

#     cms.PSet(
#         record = cms.string('L1MuTriggerScalesRcd'),
#         type = cms.string('L1MuTriggerScales'),
#         key = cms.string('L1T_Scales_20080926_startup')
#     ),

# )

# # Alternative 2:
# # For a known subsystem key (MySubsystemKey):
# # [x] Works!

process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

# xxxKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
process.L1TriggerKeyDummy.gmtKey = cms.string('test20080429')

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScaleKeysOnlineProd_cfi")
process.L1MuTriggerScaleKeysOnlineProd.subsystemLabel = cms.string('')


# # Alternative 3:
# # For a known TSC key (MyTSCKey):
# # [x] Works!

# process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
# process.L1SubsystemKeysOnline.tscKey = cms.string( 'TSC_CRUZET2_080613_GTmuon_GMTDTRPC5CSC5_CSCclosedwindow_DTTFtopbot_RPC_LUM_GCT_RCTH' )
 

# # Subclass of L1ObjectKeysOnlineProdBase.
# process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersKeysOnlineProd_cfi")
# process.L1MuGMTParametersKeysOnlineProd.subsystemLabel = cms.string('')


