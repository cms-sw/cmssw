# The following comments couldn't be translated into the new config version:

# Config file to be used to produce uncalibrated RecHits from 2006 Data
# 21/08/06 Giuseppe Della Ricca, Pietro Govoni and Alex Zabi.

import FWCore.ParameterSet.Config as cms

process = cms.Process("uncalibRecHitsProd")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#
# Ecal unpacker
process.load("EventFilter.EcalTBRawToDigi.ecalTBunpack_cfi")

# Read 2006 offline DataBase
process.load("Configuration.EcalTB.readConfiguration2006_v1_fromDB_cff")

# Reconstruction for 2006 rawData
process.load("Configuration.EcalTB.localReco2006_rawData_cff")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('rfio:INPUTFOLDER/INPUTFILE.root'),
    isBinary = cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('OUTPUTFILE.root'),
    outputCommands = cms.untracked.vstring('keep *', 
        'drop FEDRawDataCollection_*_*_*')
)

process.p = cms.Path(process.getCond*process.ecalTBunpack*process.localReco2006_rawData)
process.ep = cms.EndPath(process.out)

