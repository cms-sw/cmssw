database = "sqlite_file:HFPhase1PMTParams_V00_mc.db"
tag = "HFPhase1PMTParams_V00_mc"
inputfile = "HFPhase1PMTParams_V00_mc.bbin"

import FWCore.ParameterSet.Config as cms

process = cms.Process('HFPhase1PMTParamsDBWrite') 

process.source = cms.Source('EmptyIOVSource',
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
) 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = database
# process.CondDB.dbFormat = cms.untracked.int32(1)

# Data is tagged in the database by the "tag" parameter specified in the
# PoolDBOutputService configuration. We then check if the tag already exists.
#
# -- If the tag does not exist, a new interval of validity (IOV) for this tag
#    is created, valid till "end of time".
#
# -- If the tag already exists: the IOV of the previous data is stopped at
#    "current time" and we register new data valid from now on (currentTime
#    is the time of the current event!). 
#
# The "record" parameter should be the same in the PoolDBOutputService
# configuration and in the module which writes the object. It is basically
# used in order to just associate the record with the tag.
#
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string("HFPhase1PMTParamsRcd"),
        tag = cms.string(tag)
    ))
)

process.filereader = cms.EDAnalyzer(
    'HFPhase1PMTParamsDBWriter',
    inputFile = cms.string(inputfile),
    record = cms.string("HFPhase1PMTParamsRcd")
)

process.p = cms.Path(process.filereader)
