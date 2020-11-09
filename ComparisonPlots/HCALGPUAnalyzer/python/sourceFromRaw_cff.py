import FWCore.ParameterSet.Config as cms

# input
FastMonitoringService = cms.Service( "FastMonitoringService",
    filePerFwkStream = cms.untracked.bool( False ),
    fastMonIntervals = cms.untracked.uint32( 2 ),
    sleepTime = cms.untracked.int32( 1 )
)

EvFDaqDirector = cms.Service( "EvFDaqDirector",
    runNumber = cms.untracked.uint32( 321177 ),

    baseDir = cms.untracked.string( "tmp" ),
    buBaseDir = cms.untracked.string( "tmp" ),

    useFileBroker = cms.untracked.bool( False ),
    fileBrokerKeepAlive = cms.untracked.bool( True ),
    fileBrokerPort = cms.untracked.string( "8080" ),
    fileBrokerUseLocalLock = cms.untracked.bool( True ),
    fuLockPollInterval = cms.untracked.uint32( 2000 ),

    requireTransfersPSet = cms.untracked.bool( False ),
    selectedTransferMode = cms.untracked.string( "" ),
    mergingPset = cms.untracked.string( "" ),

    outputAdler32Recheck = cms.untracked.bool( False ),
)

source = cms.Source( "FedRawDataInputSource",
    runNumber = cms.untracked.uint32( 321177 ),
    getLSFromFilename = cms.untracked.bool(True),
    testModeNoBuilderUnit = cms.untracked.bool(False),
    verifyAdler32 = cms.untracked.bool( True ),
    verifyChecksum = cms.untracked.bool( True ),
    useL1EventID = cms.untracked.bool( False ),         # True
    alwaysStartFromfirstLS = cms.untracked.uint32( 0 ),

    eventChunkBlock = cms.untracked.uint32( 240 ),      # 32
    eventChunkSize = cms.untracked.uint32( 240),        # 32
    maxBufferedFiles = cms.untracked.uint32( 8 ),       #  2
    numBuffers = cms.untracked.uint32( 8 ),             #  2

    fileListMode = cms.untracked.bool( True ),          # False
    fileNames = cms.untracked.vstring(
        '/data/store/dalfonso/run321177_ls0142_index000000.raw',
        '/data/store/dalfonso/run321177_ls0142_index000001.raw',
        '/data/store/dalfonso/run321177_ls0142_index000002.raw',
        '/data/store/dalfonso/run321177_ls0142_index000003.raw',
        '/data/store/dalfonso/run321177_ls0142_index000004.raw',
    ),
)
