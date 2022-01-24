import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('runNumber',
                 1, #default value, int limit -3                                                                                                                              
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
options.register('transDelay',
                 0, #default value, int limit -3                                                                                                                            
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "delay in seconds for the commit of the db transaction")
options.parseArguments()

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32( options.runNumber ),
                            firstLuminosityBlock = cms.untracked.uint32( 1 ),
                            numberEventsInRun = cms.untracked.uint32( 30 ),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(3),
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(30))

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(1),
        authenticationPath = cms.untracked.string('.')
    ),
    jobName = cms.untracked.string("TestLumiBasedUpdate"),
    connect = cms.string('oracle://cms_orcoff_prep/CMS_CONDITIONS'),
    preLoadConnectionString = cms.untracked.string('frontier://FrontierPrep/CMS_CONDITIONS'),
    runNumber = cms.untracked.uint64(options.runNumber),
    lastLumiFile = cms.untracked.string('last_lumi.txt'),
    frontierKey = cms.untracked.string('test'),
    writeTransactionDelay = cms.untracked.uint32(options.transDelay),
    autoCommit = cms.untracked.bool(True),
    saveLogsOnDB = cms.untracked.bool(True),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('BeamSpot_test_updateByLumi_01'),
        timetype = cms.untracked.string('Lumi'),
        refreshTime = cms.untracked.uint32( 2 ),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    ))
)

process.mytest = cms.EDAnalyzer("LumiBasedUpdateAnalyzer",
    record = cms.untracked.string('PedestalsRcd'),
    iovSize = cms.untracked.uint32(4),
    lastLumiFile = cms.untracked.string('last_lumi.txt'),
)

process.p = cms.Path(process.mytest)



