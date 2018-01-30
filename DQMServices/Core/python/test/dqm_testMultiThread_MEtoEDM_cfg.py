import FWCore.ParameterSet.Config as cms

myWorkflow = '/My/Test/Workflow'

process = cms.Process("DQMMULTITHREAD")
process.load("DQMServices.Core.DQM_cfg")

### standard MessageLoggerConfiguration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules.append('*')
process.MessageLogger.categories.append("DQMEDAnalyzer")
process.MessageLogger.cout = cms.untracked.PSet(
   threshold = cms.untracked.string('DEBUG'),
   default = cms.untracked.PSet( limit = cms.untracked.int32(-1) ))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(50),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            firstRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('%s_MEtoEDM.root' % (myWorkflow[1:].replace('/', '__'))),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.saveByRun = cms.untracked.int32(1)
process.dqmSaver.workflow = cms.untracked.string(myWorkflow)

process.load("DQMServices.Components.DQMStoreStats_cfi")

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.dqm_multi_thread_a = DQMEDAnalyzer('DQMTestMultiThread',
                                            folder = cms.untracked.string("A_Folder/Module"),
                                            fillValue = cms.untracked.double(2.))
process.dqm_multi_thread_b = DQMEDAnalyzer('DQMTestMultiThread',
                                            folder = cms.untracked.string("B_Folder/Module"),
                                            fillValue = cms.untracked.double(3.))

process.p = cms.Path(process.dqm_multi_thread_a
                     * process.dqm_multi_thread_b
                     * process.dqmStoreStats
                     * process.dqmSaver)

process.out = cms.EndPath(process.MEtoEDMConverter*process.DQMoutput)

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32( 5 ),
    numberOfThreads = cms.untracked.uint32( 5 ),
)

# Enable MultiThread DQM
process.dqmSaver.enableMultiThread = cms.untracked.bool(True)
process.MEtoEDMConverter.enableMultiThread = cms.untracked.bool(True)


#process.Tracer = cms.Service('Tracer')
