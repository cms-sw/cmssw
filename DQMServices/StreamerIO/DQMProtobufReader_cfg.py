import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process('RECO')

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
)

# Input source
DQMProtobufReader = cms.Source("DQMProtobufReader",
    runNumber = cms.untracked.uint32(options.runNumber),
    runInputDir = cms.untracked.string(options.runInputDir),
    streamLabel = cms.untracked.string(options.streamLabel),

    delayMillis = cms.untracked.uint32(300),
    skipFirstLumis = cms.untracked.bool(False),
    endOfRunKills  = cms.untracked.bool(False),
)

process.source = DQMProtobufReader

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("cerr"),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
    ),
    #debugModules = cms.untracked.vstring('*'),
)

# Input source
#print dir(process)
#process.source = process.DQMStreamerReader

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('test_11_a_1 nevts:100'),
    name = cms.untracked.string('Applications')
)

process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )
process.endjob_step = cms.EndPath(process.endOfProcess)
process.schedule = cms.Schedule(process.endjob_step)
