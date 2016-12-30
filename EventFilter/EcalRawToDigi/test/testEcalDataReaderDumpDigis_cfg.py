# Configuration file to test the EcalDataReader. It
# converts digis, produced beforehand with EcalDataReader
# module into raw data using the EcalDigiToRaw module
# and dump the resulting raw data unsing the EcalDataReader
# module dump functionality.
#
# Original author: Ph. Gras CEA/Saclay

import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMPDIGS")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("EventFilter.EcalRawToDigi.EcalDataReader_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
# trivial conditions for ECAL Channel Status == 0
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.source = cms.Source("PoolSource",
 fileNames =
 cms.untracked.vstring('file:digi.root'),
)

import copy
process.ecalDumpRaw.dump = True
process.ecalDumpRaw.produceDigis = False
process.ecalDumpRaw.produceSrfs = False
process.ecalDumpRaw.produceTpgs = False
process.ecalDumpRaw.produceDccHeaders = False
process.ecalDumpRaw.ecalRawDataCollection = cms.InputTag("ecalDigiToRaw")

process.ecalDigiToRaw = cms.EDProducer("EcalDigiToRaw",
    InstanceEB = cms.string('ebDigis'),
    InstanceEE = cms.string('eeDigis'),
    DoEndCap = cms.untracked.bool(True),
    labelTT = cms.InputTag("ecalDataReader", ""),
    Label = cms.string('ecalDataReader'),
    labelDCCHeader = cms.InputTag('ecalDataReader'),
    debug = cms.untracked.bool(False),
#    labelEESRFlags = cms.InputTag("ecalDataReader","eeSrFlags"),
    labelEESRFlags = cms.InputTag("ecalDataReader",""),
    WriteSRFlags = cms.untracked.bool(True),
    WriteTowerBlock = cms.untracked.bool(True),
#    labelEBSRFlags = cms.InputTag("ecalDataReader","ebSrFlags"),
    labelEBSRFlags = cms.InputTag("ecalDataReader",""),
    listDCCId = cms.untracked.vint32(1, 2, 3, 4, 5,
        6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33, 34, 35,
        36, 37, 38, 39, 40,
        41, 42, 43, 44, 45,
        46, 47, 48, 49, 50,
        51, 52, 53, 54),
    WriteTCCBlock = cms.untracked.bool(True),
    DoBarrel = cms.untracked.bool(True)
)

process.path = cms.Path(process.ecalDigiToRaw*process.ecalDumpRaw)
