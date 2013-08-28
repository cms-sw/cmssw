import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDBToFile")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(100000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT_V14P::All"

# process.calibDB = cms.ESSource("PoolDBESSource",
#     process.CondDBSetup,
#     authenticationMethod = cms.untracked.uint32(0),
#     toGet = cms.VPSet(cms.PSet(
#         # VDrift
#         #record = cms.string("DTMtimeRcd"),
#         #tag = cms.string("DT_vDrift_CRAFT_V02_offline")
#         # TZero
#         #record = cms.string("DTT0Rcd" ),
#         #tag = cms.string("t0"),
#         # TTrig
#         record = cms.string('DTTtrigRcd'),
#         tag = cms.string('ttrig')
#     )),
#     connect = cms.string('frontier://FrontierPrep/CMS_COND_31X_All')
# )

process.dumpToFile = cms.EDAnalyzer("DumpDBToFile",
    #Choose what database you want to write
    #untracked string dbToDump = "VDriftDB"
    #untracked string dbToDump = "TZeroDB"
    dbToDump = cms.untracked.string('TTrigDB'),
    dbLabel = cms.untracked.string(''),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        # VDrift & TTrig
        calibConstGranularity = cms.untracked.string('bySL')
    ),
    outputFileName = cms.untracked.string('ttrig2.txt')
)

process.p = cms.Path(process.dumpToFile)


