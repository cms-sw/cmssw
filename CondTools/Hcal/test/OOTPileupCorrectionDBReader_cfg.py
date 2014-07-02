database = "sqlite_file:testOOTPileupCorrection.db"
tag = "test"
outputfile = "testOOTPileupCorrection_dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('OOTPileupCorrectionDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database
process.CondDB.dbFormat = cms.untracked.int32(1)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalOOTPileupCorrectionRcd"),
        tag = cms.string(tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'OOTPileupCorrectionDBReader',
    outputFile = cms.string(outputfile),
    dumpMetadata = cms.untracked.bool(True)
)

process.p = cms.Path(process.dumper)
