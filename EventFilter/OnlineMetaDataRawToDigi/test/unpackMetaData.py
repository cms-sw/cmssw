import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source(
    "NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
        #"/store/user/mommsen/testData/run309216_ls0038_streamExpressCosmics_StorageManager.dat"  #version 1 data format
        "/store/user/mommsen/testData/run309369_ls0004_streamExpressCosmics_StorageManager.dat" #version 2 data format
        )
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

process.load('EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi')

process.tcdsRaw = cms.EDProducer('TcdsRawToDigi')

process.metaData = cms.EDProducer('OnlineMetaDataRawToDigi')

process.dumpRaw = cms.EDAnalyzer("DumpFEDRawDataProduct",
    feds = cms.untracked.vint32 (735,1022,1024),
    label = cms.untracked.string("rawDataCollector"),
    dumpPayload = cms.untracked.bool( True )
    )

process.path = cms.Path(
    process.tcdsRaw
    +process.metaData
    +process.dumpRaw
    )

process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *"),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('path')
    ),
    fileName = cms.untracked.string('test.root')
    )

process.out = cms.EndPath(
    process.output
    )
