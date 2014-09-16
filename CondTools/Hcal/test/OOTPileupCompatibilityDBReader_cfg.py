database = "sqlite_file:compatOOTPileupCorrection.db"
tag = "test"
outputfile = "testOOTPileupCorrection_dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('OOTPileupCompatibilityDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalOOTPileupCompatibilityRcd"),
        tag = cms.string(tag)
    ))
)

process.ootpileupesproducer = cms.ESProducer(
    'OOTPileupDBCompatibilityESProducer'
)

process.dumper = cms.EDAnalyzer(
    'OOTPileupCompatibilityDBReader',
    outputFile = cms.string(outputfile)
)

process.p = cms.Path(process.dumper)
