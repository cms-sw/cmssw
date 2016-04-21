import FWCore.ParameterSet.Config as cms

process = cms.Process("TotemIntegratedRawDataTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/j/jkaspar/public/run268608_ls0001_streamA_StorageManager.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# raw-to-digi conversion
process.load('CondFormats.TotemReadoutObjects.TotemDAQMappingESSourceXML_cfi')
process.TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/TotemReadoutObjects/xml/ctpps_210_mapping.xml")

# in the emulated data the trigger block contains non-sense
#process.load("EventFilter.TotemRawToDigi.totemTriggerRawToDigi_cfi")
#process.totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")
#process.totemTriggerRawToDigi.fedId = 577

process.load('EventFilter.TotemRawToDigi.totemRPRawToDigi_cfi')
process.totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")
process.totemRPRawToDigi.fedIds = cms.vuint32(578, 579, 580) # in the emulated data one OptoRx was not functional
process.totemRPRawToDigi.RawToDigi.printErrorSummary = 1
process.totemRPRawToDigi.RawToDigi.printUnknownFrameSummary = 1

# execution configuration
process.p = cms.Path(
    #process.totemTriggerRawToDigi *
    process.totemRPRawToDigi
)
