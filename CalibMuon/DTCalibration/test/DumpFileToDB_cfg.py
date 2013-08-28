import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpFileToDB")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


# VDrift, TTrig, TZero, Noise or channels Map into DB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:ttrig_W-1S4NewTW.db"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("DTTtrigRcd"),
                                                                     tag = cms.string("ttrig_W-1S4NewTW"))))


#Module to dump a file into a DB
process.dumpToDB = cms.EDAnalyzer("DumpFileToDB",
                                calibFileConfig = cms.untracked.PSet(calibConstFileName = cms.untracked.string("ttrig2.txt"),
                                                                     calibConstGranularity = cms.untracked.string('bySL'),
                                                                     nFields = cms.untracked.int32(4)
                                                                     # VDrift & TTrig
                                                                     #untracked string calibConstGranularity = "bySL"
                                                                     #untracked int32 nFields = 4
                                                                     # TZero
                                                                     #untracked string calibConstGranularity = "byWire"
                                                                     #untracked int32 nFields = 6
                                                                     # Noise
                                                                     #untracked string calibConstGranularity = "byWire"
                                                                     #untracked int32 nFields = 7
                                                                     # Dead
                                                                     #untracked string calibConstGranularity = "byWire"
                                                                     #untracked int32 nFields = 7
                                                                     # No parameters required for ChannelDB
                                                                     ),
                                #Choose what database you want to write
                                #untracked string dbToDump = "VDriftDB"
                                #untracked string dbToDump = "TZeroDB"
                                #untracked string dbToDump = "TTrigDB"
                                #untracked string dbToDump = "NoiseDB"
                                #untracked string dbToDump = "DeadDB"
                                dbToDump = cms.untracked.string('TTrigDB'))
                                

process.p = cms.Path(process.dumpToDB)
    

