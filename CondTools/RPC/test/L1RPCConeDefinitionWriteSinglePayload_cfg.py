import FWCore.ParameterSet.Config as cms

# This is to write a single payload of L1RPCConeDefinition to the DB using the L1 O2O.
# CondTools/L1Trigger/test/init_cfg.py must be run first to initialize the
# L1TriggerKeyList in the DB, then L1RPCConeDefinition payloads can be written one by one.

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate L1TriggerKey - here define the RPC key
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
    record = cms.string('L1RPCConeDefinitionRcd'),
    type = cms.string('L1RPCConeDefinition'),
    key = cms.string('DEFAULT')
#    key = cms.string('COSMIC')
))

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.orcon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_IDEAL')
    ))
)
process.es_prefer_orcon = cms.ESPrefer("PoolDBESSource","orcon")
process.orcon.connect = cms.string('sqlite_file:l1config.db')
process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')

# Generate L1RPCConeDefinition object - here define the payload associated to the RPC key
process.RPCConeDefi = cms.ESProducer("L1RPCConeDefinitionProducer",
    rollConnT_14_4 = cms.vint32(-1, -1, -1),
    rollConnT_14_5 = cms.vint32(-1, -1, -1),
    rollConnT_14_0 = cms.vint32(13, 14, -1),
    rollConnT_14_1 = cms.vint32(13, -1, -1),
    rollConnT_14_2 = cms.vint32(14, 15, -1),
    rollConnT_14_3 = cms.vint32(15, 16, -1),
    rollConnT_12_4 = cms.vint32(-1, -1, -1),
    rollConnT_12_5 = cms.vint32(-1, -1, -1),
    rollConnT_12_2 = cms.vint32(12, 13, -1),
    rollConnT_12_3 = cms.vint32(13, 14, -1),
    rollConnT_12_0 = cms.vint32(10, 11, -1),
    rollConnT_12_1 = cms.vint32(11, -1, -1),
    lpSizeTower0 = cms.vint32(72, 56, 8, 40, 40, 
        24),
    rollConnLP_8_2 = cms.vint32(3, -5, 0),
    # HwPlane 5
    rollConnT_0_4 = cms.vint32(-1, -1, -1),
    rollConnT_9_4 = cms.vint32(-1, -1, -1),
    rollConnLP_13_4 = cms.vint32(0, 0, 0),
    #
    #
    #
    # m_mrtow table
    # rollConn_[RollNo]_[hwPlane]
    # HwPlane 1
    rollConnT_0_0 = cms.vint32(-1, -1, -1),
    rollConnLP_10_1 = cms.vint32(2, 0, 0),
    rollConnT_9_0 = cms.vint32(7, 8, -1),
    lpSizeTower14 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    lpSizeTower15 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    lpSizeTower16 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    lpSizeTower10 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    lpSizeTower11 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    lpSizeTower12 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    lpSizeTower13 = cms.vint32(72, 8, 40, 24, 0, 
        0),
    rollConnT_9_2 = cms.vint32(9, -9, 10),
    rollConnT_6_2 = cms.vint32(-1, -1, -1),
    towerBeg = cms.int32(0),
    rollConnT_6_3 = cms.vint32(-1, -1, -1),
    rollConnT_6_0 = cms.vint32(-1, -1, -1),
    rollConnT_16_2 = cms.vint32(-1, -1, -1),
    rollConnT_16_3 = cms.vint32(-1, -1, -1),
    rollConnT_16_0 = cms.vint32(15, 16, -1),
    rollConnT_16_1 = cms.vint32(15, -1, -1),
    rollConnLP_3_0 = cms.vint32(0, 0, 0),
    rollConnT_16_4 = cms.vint32(-1, -1, -1),
    rollConnT_16_5 = cms.vint32(-1, -1, -1),
    rollConnT_8_4 = cms.vint32(-1, -1, -1),
    rollConnT_15_4 = cms.vint32(-1, -1, -1),
    rollConnLP_6_5 = cms.vint32(4, 0, 0),
    rollConnLP_6_4 = cms.vint32(0, 0, 0),
    rollConnLP_6_3 = cms.vint32(0, 0, 0),
    rollConnLP_6_2 = cms.vint32(0, 0, 0),
    rollConnT_8_2 = cms.vint32(9, -9, -1),
    rollConnLP_6_0 = cms.vint32(0, 0, 0),
    # HwPlane 4
    rollConnT_0_3 = cms.vint32(-1, -1, -1),
    rollConnT_13_5 = cms.vint32(-1, -1, -1),
    rollConnT_13_4 = cms.vint32(-1, -1, -1),
    rollConnT_13_3 = cms.vint32(14, 15, -1),
    rollConnT_13_2 = cms.vint32(13, 14, -1),
    rollConnT_13_1 = cms.vint32(12, -1, -1),
    rollConnT_13_0 = cms.vint32(11, 12, -1),
    rollConnLP_4_1 = cms.vint32(3, 0, 0),
    rollConnLP_4_0 = cms.vint32(1, 1, -5),
    rollConnLP_4_3 = cms.vint32(6, 6, 0),
    rollConnLP_4_2 = cms.vint32(5, 5, 5),
    rollConnLP_4_5 = cms.vint32(4, 4, 0),
    rollConnLP_4_4 = cms.vint32(2, 2, 0),
    hwPlaneBeg = cms.int32(0),
    rollConnLP_12_5 = cms.vint32(0, 0, 0),
    rollConnT_10_1 = cms.vint32(9, -1, -1),
    rollConnT_4_0 = cms.vint32(4, 5, -6),
    rollConnT_4_1 = cms.vint32(4, -1, -1),
    rollConnT_4_2 = cms.vint32(2, 3, 4),
    rollConnT_4_3 = cms.vint32(2, 3, -1),
    rollConnT_4_4 = cms.vint32(4, 5, -1),
    rollConnT_4_5 = cms.vint32(3, 4, -1),
    rollConnLP_8_5 = cms.vint32(0, 0, 0),
    rollConnLP_8_4 = cms.vint32(0, 0, 0),
    rollConnT_6_4 = cms.vint32(-1, -1, -1),
    rollConnT_6_5 = cms.vint32(6, -1, -1),
    rollConnLP_8_1 = cms.vint32(0, 0, 0),
    rollConnLP_8_0 = cms.vint32(0, 5, 0),
    rollConnLP_8_3 = cms.vint32(4, 0, 0),
    rollConnT_6_1 = cms.vint32(-1, -1, -1),
    rollConnT_2_2 = cms.vint32(1, 2, -1),
    rollConnT_2_3 = cms.vint32(1, 2, -1),
    rollConnLP_16_0 = cms.vint32(1, 1, 0),
    rollConnT_2_0 = cms.vint32(2, 3, 4),
    rollConnT_15_3 = cms.vint32(16, -1, -1),
    rollConnT_2_1 = cms.vint32(2, -1, -1),
    rollConnT_8_5 = cms.vint32(-1, -1, -1),
    rollConnT_7_2 = cms.vint32(5, 6, -1),
    rollConnLP_2_3 = cms.vint32(6, 6, 0),
    rollConnLP_2_2 = cms.vint32(5, 5, 0),
    rollConnLP_2_1 = cms.vint32(3, 0, 0),
    rollConnLP_2_0 = cms.vint32(1, 1, 1),
    rollConnLP_2_5 = cms.vint32(4, 4, 0),
    rollConnLP_2_4 = cms.vint32(2, 2, 2),
    rollConnT_11_1 = cms.vint32(10, -1, -1),
    rollConnT_11_0 = cms.vint32(10, -1, -1),
    rollConnT_11_3 = cms.vint32(12, 13, -1),
    rollConnT_11_2 = cms.vint32(11, 12, -1),
    rollConnT_11_5 = cms.vint32(-1, -1, -1),
    rollConnT_11_4 = cms.vint32(-1, -1, -1),
    rollConnT_17_3 = cms.vint32(-1, -1, -1),
    rollConnT_17_2 = cms.vint32(-1, -1, -1),
    rollConnT_17_1 = cms.vint32(16, -1, -1),
    rollConnT_17_0 = cms.vint32(16, -1, -1),
    # HwPlane 2
    rollConnLP_0_1 = cms.vint32(3, 0, 0),
    #
    #
    #
    #m_mrlogp table
    # HwPlane 1
    rollConnLP_0_0 = cms.vint32(0, 0, 0),
    rollConnT_17_5 = cms.vint32(-1, -1, -1),
    rollConnT_17_4 = cms.vint32(-1, -1, -1),
    rollConnT_15_5 = cms.vint32(-1, -1, -1),
    rollConnLP_14_5 = cms.vint32(0, 0, 0),
    rollConnLP_14_4 = cms.vint32(0, 0, 0),
    rollConnLP_14_3 = cms.vint32(4, 4, 0),
    rollConnLP_14_2 = cms.vint32(3, 3, 0),
    rollConnLP_14_1 = cms.vint32(2, 0, 0),
    rollConnLP_14_0 = cms.vint32(1, 1, 0),
    rollConnT_9_5 = cms.vint32(-1, -1, -1),
    # HwPlane 6
    rollConnT_0_5 = cms.vint32(-1, -1, -1),
    rollConnLP_7_4 = cms.vint32(2, 2, 0),
    rollConnLP_7_5 = cms.vint32(4, 0, 0),
    rollConnT_9_1 = cms.vint32(8, -1, -1),
    # HwPlane 2
    rollConnT_0_1 = cms.vint32(0, -1, -1),
    # HwPlane 3
    rollConnT_0_2 = cms.vint32(-1, -1, -1),
    rollConnLP_7_1 = cms.vint32(3, -3, 0),
    lpSizeTower7 = cms.vint32(72, 56, 40, 8, 24, 
        0),
    rollConnT_7_1 = cms.vint32(7, -7, -1),
    rollConnLP_16_1 = cms.vint32(2, 0, 0),
    rollConnT_15_1 = cms.vint32(14, -1, -1),
    rollConnLP_16_3 = cms.vint32(0, 0, 0),
    rollConnLP_16_2 = cms.vint32(0, 0, 0),
    rollConnLP_16_5 = cms.vint32(0, 0, 0),
    rollConnLP_16_4 = cms.vint32(0, 0, 0),
    rollConnT_15_0 = cms.vint32(14, 15, -1),
    rollConnLP_5_0 = cms.vint32(1, 1, 1),
    rollConnLP_5_1 = cms.vint32(3, -3, 3),
    rollConnLP_5_2 = cms.vint32(5, 5, 0),
    rollConnLP_5_3 = cms.vint32(6, 0, 0),
    rollConnLP_5_4 = cms.vint32(2, 2, 2),
    rollConnLP_5_5 = cms.vint32(4, 0, 0),
    rollConnT_2_4 = cms.vint32(2, 3, 4),
    rollConnT_2_5 = cms.vint32(2, 3, -1),
    rollConnT_8_3 = cms.vint32(10, -1, -1),
    rollConnT_3_1 = cms.vint32(3, -1, -1),
    rollConnT_5_1 = cms.vint32(5, -6, 6),
    rollConnT_5_0 = cms.vint32(6, 7, 8),
    rollConnT_5_3 = cms.vint32(4, -1, -1),
    rollConnT_5_2 = cms.vint32(4, 5, -1),
    rollConnT_5_5 = cms.vint32(5, -1, -1),
    rollConnT_5_4 = cms.vint32(5, 6, 7),
    rollConnT_1_4 = cms.vint32(0, 1, -2),
    rollEnd = cms.int32(17),
    hwPlaneEnd = cms.int32(5),
    rollConnLP_10_3 = cms.vint32(4, 4, 0),
    rollConnLP_10_2 = cms.vint32(3, 3, 0),
    rollConnT_9_3 = cms.vint32(10, 11, -1),
    rollConnLP_10_0 = cms.vint32(3, 0, 0),
    rollConnLP_10_5 = cms.vint32(0, 0, 0),
    rollConnLP_10_4 = cms.vint32(0, 0, 0),
    rollConnLP_3_2 = cms.vint32(0, 0, 0),
    rollConnLP_3_3 = cms.vint32(0, 0, 0),
    rollConnT_15_2 = cms.vint32(15, 16, -1),
    rollConnLP_3_1 = cms.vint32(3, 0, 0),
    rollConnLP_6_1 = cms.vint32(0, 0, 0),
    rollConnLP_3_4 = cms.vint32(0, 0, 0),
    rollConnLP_3_5 = cms.vint32(0, 0, 0),
    rollConnT_10_0 = cms.vint32(8, -1, -1),
    rollConnLP_12_4 = cms.vint32(0, 0, 0),
    rollConnT_10_2 = cms.vint32(10, 11, -1),
    rollConnT_10_3 = cms.vint32(11, 12, -1),
    rollConnLP_12_1 = cms.vint32(2, 0, 0),
    rollConnLP_12_0 = cms.vint32(1, 1, 0),
    rollConnLP_12_3 = cms.vint32(4, 4, 0),
    rollConnLP_12_2 = cms.vint32(3, 3, 0),
    rollConnLP_1_4 = cms.vint32(2, 2, -2),
    rollConnLP_1_5 = cms.vint32(4, 4, 0),
    rollBeg = cms.int32(0),
    rollConnLP_1_0 = cms.vint32(1, 1, -1),
    rollConnLP_1_1 = cms.vint32(3, 0, 0),
    rollConnLP_1_2 = cms.vint32(5, 5, 0),
    rollConnLP_1_3 = cms.vint32(6, 6, 0),
    rollConnLP_9_4 = cms.vint32(0, 0, 0),
    rollConnLP_9_5 = cms.vint32(0, 0, 0),
    rollConnT_7_5 = cms.vint32(7, -1, -1),
    rollConnT_7_4 = cms.vint32(7, 8, -1),
    rollConnLP_9_0 = cms.vint32(5, 3, 0),
    rollConnLP_9_1 = cms.vint32(4, 0, 0),
    rollConnLP_9_2 = cms.vint32(3, -5, 3),
    rollConnT_7_0 = cms.vint32(8, 9, -1),
    rollConnLP_13_0 = cms.vint32(1, 1, 0),
    lpSizeTower3 = cms.vint32(72, 56, 8, 40, 40, 
        24),
    rollConnT_1_5 = cms.vint32(0, 1, -1),
    rollConnLP_17_1 = cms.vint32(2, 0, 0),
    rollConnLP_17_2 = cms.vint32(0, 0, 0),
    rollConnLP_17_3 = cms.vint32(0, 0, 0),
    rollConnLP_9_3 = cms.vint32(4, 4, 0),
    rollConnT_1_0 = cms.vint32(0, 1, -2),
    rollConnT_1_3 = cms.vint32(0, 1, -1),
    rollConnT_1_2 = cms.vint32(0, 1, -1),
    rollConnT_1_1 = cms.vint32(1, -1, -1),
    lpSizeTower1 = cms.vint32(72, 56, 8, 40, 40, 
        24),
    rollConnT_7_3 = cms.vint32(4, 5, -1),
    # HwPlane 6
    rollConnLP_0_5 = cms.vint32(0, 0, 0),
    rollConnLP_7_2 = cms.vint32(5, 5, 0),
    # HwPlane 5
    rollConnLP_0_4 = cms.vint32(0, 0, 0),
    rollConnT_3_3 = cms.vint32(-1, -1, -1),
    rollConnT_3_2 = cms.vint32(-1, -1, -1),
    rollConnLP_17_4 = cms.vint32(0, 0, 0),
    rollConnT_3_0 = cms.vint32(-1, -1, -1),
    rollConnLP_17_0 = cms.vint32(1, 0, 0),
    rollConnT_3_5 = cms.vint32(-1, -1, -1),
    rollConnT_3_4 = cms.vint32(-1, -1, -1),
    rollConnT_10_4 = cms.vint32(-1, -1, -1),
    rollConnT_8_0 = cms.vint32(-1, 7, -1),
    rollConnLP_7_3 = cms.vint32(6, 6, 0),
    # HwPlane 4
    rollConnLP_0_3 = cms.vint32(0, 0, 0),
    rollConnLP_17_5 = cms.vint32(0, 0, 0),
    # HwPlane 3
    rollConnLP_0_2 = cms.vint32(0, 0, 0),
    rollConnT_10_5 = cms.vint32(-1, -1, -1),
    rollConnLP_11_2 = cms.vint32(3, 3, 0),
    rollConnLP_11_3 = cms.vint32(4, 4, 0),
    rollConnLP_11_0 = cms.vint32(1, 0, 0),
    rollConnLP_11_1 = cms.vint32(2, 0, 0),
    rollConnLP_11_4 = cms.vint32(0, 0, 0),
    rollConnLP_11_5 = cms.vint32(0, 0, 0),
    rollConnT_8_1 = cms.vint32(-1, -1, -1),
    rollConnLP_7_0 = cms.vint32(1, 1, 0),
    lpSizeTower8 = cms.vint32(72, 24, 40, 8, 0, 
        0),
    lpSizeTower9 = cms.vint32(72, 8, 40, 0, 0, 
        0),
    #vint32 lpSizeTower5 = {72, 56,  8, 8, 40, 24}  // old CMSSW bug compatible
    lpSizeTower6 = cms.vint32(56, 72, 40, 8, 24, 
        0),
    rollConnLP_13_5 = cms.vint32(0, 0, 0),
    lpSizeTower4 = cms.vint32(72, 56, 8, 40, 40, 
        24),
    #vint32 lpSizeTower5 = {72, 56,  8, 40, 40, 24} // Bug
    lpSizeTower5 = cms.vint32(72, 56, 40, 8, 40, 
        24),
    lpSizeTower2 = cms.vint32(72, 56, 8, 40, 40, 
        24),
    rollConnLP_13_1 = cms.vint32(2, 0, 0),
    rollConnLP_13_2 = cms.vint32(3, 3, 0),
    rollConnLP_13_3 = cms.vint32(4, 4, 0),
    towerEnd = cms.int32(16),
    rollConnLP_15_4 = cms.vint32(0, 0, 0),
    rollConnLP_15_5 = cms.vint32(0, 0, 0),
    rollConnLP_15_2 = cms.vint32(3, 3, 0),
    rollConnLP_15_3 = cms.vint32(4, 0, 0),
    rollConnLP_15_0 = cms.vint32(1, 1, 0),
    rollConnLP_15_1 = cms.vint32(2, 0, 0)
)
RPCConeDefSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCConeDefinitionRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cff")
process.L1CondDBPayloadWriter.writeL1TriggerKey = cms.bool(False)
process.L1CondDBPayloadWriter.L1TriggerKeyListTag = cms.string('L1TriggerKeyList_IDEAL')
process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')
process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')

# Use highest possible run number so we always get the latest version
# of L1TriggerKeyList.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(4294967295),
    lastValue = cms.uint64(4294967295),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBPayloadWriter)

