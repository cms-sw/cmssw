import FWCore.ParameterSet.Config as cms
##################################################################
# Print the contents of the RawDataBuffer EDProduct (corresponding to DAQ raw data)
##################################################################
process = cms.Process("DUMP")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:BtlDigiToRaw.root")
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.DumpBTLRawDataBuffer = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

process.dumpBTLRawDataBuffer = cms.EDAnalyzer("DumpBTLRawDataBuffer",
                                              rawDataBufferTag = cms.InputTag("btlDigiToRaw","","BTLPACKER"),
)

process.path = cms.Path(process.dumpBTLRawDataBuffer)
