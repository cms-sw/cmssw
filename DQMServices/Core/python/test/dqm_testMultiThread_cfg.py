import FWCore.ParameterSet.Config as cms

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

process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(50),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            firstRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.saveByRun = cms.untracked.int32(1)
process.dqmSaver.workflow = cms.untracked.string("/My/Test/Workflow")
process.dqmSaver.enableMultiThread = cms.untracked.bool(True)

process.load("DQMServices.Components.DQMStoreStats_cfi")

process.dqm_multi_thread_a = cms.EDAnalyzer("DQMTestMultiThread",
                                            folder = cms.untracked.string("A_Folder/Module"))
process.dqm_multi_thread_b = cms.EDAnalyzer("DQMTestMultiThread",
                                            folder = cms.untracked.string("B_Folder/Module"))

process.p = cms.Path(process.dqm_multi_thread_a
                     * process.dqm_multi_thread_b
                     * process.dqmStoreStats
                     * process.dqmSaver)

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32( 5 ),
    numberOfThreads = cms.untracked.uint32( 5 ),
)


#process.Tracer = cms.Service('Tracer')
