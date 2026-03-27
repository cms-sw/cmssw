database = "sqlite_file:hcalPulseMap.db"
tag = "test"
outputfile = "hcalPulseMap_dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('HcalInterpolatedPulseMapDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HcalInterpolatedPulseMapRcd"),
        tag = cms.string(tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'HcalInterpolatedPulseMapDBReader',
    outputFile = cms.string(outputfile)
)

process.p = cms.Path(process.dumper)
