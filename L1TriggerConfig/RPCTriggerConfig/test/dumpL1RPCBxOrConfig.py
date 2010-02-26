import FWCore.ParameterSet.Config as cms

process = cms.Process("dump")

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCBxOrConfig_cff")
process.l1RPCBxOrConfig.firstBX=cms.int32(-2)
process.l1RPCBxOrConfig.lastBX=cms.int32(0)

#useGlobalTag = 'IDEAL_31X'
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = useGlobalTag + '::All'

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


process.write = cms.EDAnalyzer("DumpL1RPCBxOrConfig",
)


process.p1 = cms.Path(process.write)
