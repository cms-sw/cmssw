database = "sqlite_file:HFPhase1PMTParams_V00_mc.db"
tag = "HFPhase1PMTParams_V00_mc"
outputfile = "dbread.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('HFPhase1PMTParamsDBRead')

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database
# process.CondDB.dbFormat = cms.untracked.int32(1)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("HFPhase1PMTParamsRcd"),
        tag = cms.string(tag)
    ))
)

process.dumper = cms.EDAnalyzer(
    'HFPhase1PMTParamsDBReader',
    outputFile = cms.string(outputfile)
)

process.p = cms.Path(process.dumper)
