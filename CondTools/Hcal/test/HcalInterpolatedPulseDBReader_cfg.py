database = "sqlite_file:hcalPulse.db"
tag = "test"
outputfile = "hcalPulse_dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('HcalInterpolatedPulseDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalInterpolatedPulseCollRcd"),
        tag = cms.string(tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'HcalInterpolatedPulseDBReader',
    outputFile = cms.string(outputfile)
)

process.p = cms.Path(process.dumper)
