database = "sqlite_file:HBHENegativeEFilter_V00_data.db"
tag = "HBHENegativeEFilter_V00_data"
outputfile = "dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('HBHENegativeEFilterDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database
# process.CondDB.dbFormat = cms.untracked.int32(1)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HBHENegativeEFilterRcd"),
        tag = cms.string(tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'HBHENegativeEFilterDBReader',
    outputFile = cms.string(outputfile)
)

process.p = cms.Path(process.dumper)
