import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpFileToDB")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(300000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.GlobalTag.globaltag = "GR10_P_V5::All"
#process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')


# process.calibDB = cms.ESSource("PoolDBESSource",
#      process.CondDBSetup,
#      authenticationMethod = cms.untracked.uint32(0),
#      toGet = cms.VPSet(cms.PSet(
#          # VDrift
#          #record = cms.string("DTMtimeRcd"),
#          #tag = cms.string("DT_vDrift_CRAFT_V02_offline")
#          # TZero
#          #record = cms.string("DTT0Rcd" ),
#          #tag = cms.string("t0"),
#          # TTrig
#          record = cms.string('DTTtrigRcd'),
#          tag = cms.string('ttrig_test')
#      )),
#      connect = cms.string('sqlite_file:ttrig_test.db')
#  )

# VDrift, TTrig, TZero, Noise or channels Map into DB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:ttrignew.db"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("DTTtrigRcd"),
                                                                     tag = cms.string("ttrig"))))


#Module to dump a file into a DB
process.dumpToDB = cms.EDFilter("DumpFileToDB",
                                differentialMode = cms.untracked.bool(True),
                                calibFileConfig = cms.untracked.PSet(calibConstFileName = cms.untracked.string("ttrig_prompt.txt"),
                                                                     calibConstGranularity = cms.untracked.string('bySL'),
                                                                     nFields = cms.untracked.int32(5)
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
    

