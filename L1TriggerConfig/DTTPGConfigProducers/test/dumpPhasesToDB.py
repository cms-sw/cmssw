import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTTrigSynchCosmicsClient")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:emulator_phase_map_fromfile.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTPGParametersRcd'),
        tag = cms.string('emulatorPhases')
    ))
)

process.dtTPGParamWriter = cms.EDAnalyzer("DTTPGParamsWriter",
     debug = cms.untracked.bool(True),
     inputFile = cms.untracked.string("phases_file.txt")
)

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTLocalTriggerSynchTest'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              DTLocalTriggerSynchTest = cms.untracked.PSet(
                                                                                   limit = cms.untracked.int32(-1))
                                                              )
                                    )

process.clientPath = cms.Path(process.dtTPGParamWriter)
