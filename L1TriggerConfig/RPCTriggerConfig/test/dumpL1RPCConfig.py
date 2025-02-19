import FWCore.ParameterSet.Config as cms

process = cms.Process("dump")

# process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")

useGlobalTag = 'IDEAL_31X'
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = useGlobalTag + '::All'

#process.MessageLogger = cms.Service("MessageLogger",
#    log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
#    debugModules = cms.untracked.vstring("*"),
#    destinations = cms.untracked.vstring('logDumpL1RPCConfig')
#)


process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['*']
process.MessageLogger.cout = cms.untracked.PSet(
    threshold=cms.untracked.string('DEBUG'),
    #threshold = cms.untracked.string('INFO'),
    #threshold = cms.untracked.string('ERROR'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    default = cms.untracked.PSet( 
        limit=cms.untracked.int32(-1)  
    )
)



process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.write = cms.EDAnalyzer("DumpL1RPCConfig",
)


process.p1 = cms.Path(process.write)
