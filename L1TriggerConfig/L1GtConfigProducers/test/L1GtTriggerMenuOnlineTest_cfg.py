from __future__ import print_function
# cfg file to test the online producer of L1GtTriggerMenuRcd

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")

###################### user choices ######################

useKey = 'ObjectKey'
#useKey = 'SubsystemKey'
#useKey = 'TscKey'

###################### end user choices ###################

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)


# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuOnline_cfi")

if (useKey == 'ObjectKey') : 
    process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
    process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
        record = cms.string('L1GtTriggerMenuRcd'),
        type = cms.string('L1GtTriggerMenu'),
        key = cms.string('L1Menu_Commissioning2009_v7/L1T_Scales_20080926_startup/Imp0/0x100f')
        ))
    
elif (useKey == 'SubsystemKey') :
    process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
    process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
    process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

    # xxxKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
    process.L1TriggerKeyDummy.gtKey = cms.string('gt_2009_test_1')

    # Subclass of L1ObjectKeysOnlineProdBase.
    process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTscObjectKeysOnline_cfi")
    process.l1GtTscObjectKeysOnline.systemLabel = cms.string('')

elif (useKey == 'TscKey') :
    # TSC key
    process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
    #process.L1SubsystemKeysOnline.tscKey = cms.string( 'TSC_000618_090304_MIDWEEK2008_GTgt20090bst30_GMTstartup3_DTTF_DT_MI' )
    process.L1SubsystemKeysOnline.tscKey = \
        cms.string( 'TSC_000990_090723_CRAFT_GTgt200911_GMTsynctf02ro3rpc2_GCT_RCT_DTTF_CSCTF_HCAL_DT_RPC_MI')
    
    # Subclass of L1ObjectKeysOnlineProdBase.
    process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTscObjectKeysOnline_cfi")
    process.l1GtTscObjectKeysOnline.systemLabel = cms.string('')

else :
    print('Error: no such key type ', useKey)  
    sys.exit()



process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1GtTriggerMenuRcd'),
   data = cms.vstring('L1GtTriggerMenu')
   )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter)

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

