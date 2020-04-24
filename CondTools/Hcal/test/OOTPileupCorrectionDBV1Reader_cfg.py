database = "sqlite_file:mcOOTPileupCorrection_v1.db"
tag = "test"
outputfile = "mcOOTPileupCorrection_v1_dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('OOTPileupCorrectionDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database
# process.CondDB.dbFormat = cms.untracked.int32(1)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalOOTPileupCorrectionMapCollRcd"),
        tag = cms.string(tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'OOTPileupCorrectionDBV1Reader',
    outputFile = cms.string(outputfile)
)

process.p = cms.Path(process.dumper)
