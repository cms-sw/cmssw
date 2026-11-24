# Occupancy of the BTL raw data (channels per event, active chips, channels per chip and per FED),
# read from a file with the BTL RawDataBuffer.
import FWCore.ParameterSet.Config as cms

process = cms.Process("BTLOCCUPANCY")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
process.MessageLogger.cerr.BTLRawOccupancyAnalyzer = cms.untracked.PSet(limit = cms.untracked.int32(-1))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #"/store/user/malberti/MTD/DPG/BTLUnpacker/TTbar_PU200_D128_Aging1000/skim/BTLRaw_TTbar_PU200_D128_Aging1000_100evts.root")
        "file:BTLRaw.root")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("BTLRawOccupancy.root")
)

process.btlRawOccupancy = cms.EDAnalyzer("BTLRawOccupancyAnalyzer",
    rawDataBufferTag = cms.InputTag("btlDigiToRaw")
)

process.path = cms.Path(process.btlRawOccupancy)
