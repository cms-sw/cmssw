
import FWCore.ParameterSet.Config as cms
process = cms.Process('HARVESTING')

process.Tracer = cms.Service("Tracer")

process.load('Configuration.StandardSequences.Services_cff')

process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(''),
    reScope = cms.untracked.string("RUN")
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    TryToContinue = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
 
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)
process.out = cms.OutputModule(
  "DQMRootOutputModule",
   fileName = cms.untracked.string("harvesting_out.root"),
   outputCommands = cms.untracked.vstring(
     'keep *'
   )
)      
process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    convention = cms.untracked.string('Offline'),
    fileFormat = cms.untracked.string('ROOT'),
    producer = cms.untracked.string('DQM'),
    workflow = cms.untracked.string('/A/B/C'),
    dirName = cms.untracked.string('.'),
)
process.o = cms.EndPath(process.out + process.dqmSaver)

process.source.fileNames = cms.untracked.vstring(
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/3C5DD0BD-4370-AD40-BF5C-2FDCE02A327A.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/E78D6DAD-556D-014F-84D4-B244EDE72106.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/81CAE175-9616-954E-B0F3-2C6E82BEBCC9.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/046EF9CC-812A-D441-9C11-3AF6F3D345DB.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/649FC63D-1924-724F-BC17-22626DFA636E.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/9604B80E-2AF1-7840-A8C0-1F689ECC6694.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/BC3E3EE5-4DF0-2A4F-BE3A-DDB72741E8D7.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/C76145B9-E43F-6F49-A3F8-0C8E567A27E2.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/8759972F-8155-6443-AA7A-CFC344539084.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/02342C8D-AA50-BB43-A232-AED595EBAC38.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/69D71662-C61D-C746-BF11-AB0FD4A1079A.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/D6BF8E7C-13AF-3D47-99E7-22C2E044E568.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/838E45E8-4A96-184B-A473-B4B3137EC215.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/E4DBF9BB-C9DC-FA44-9BE1-84CA9AD257D2.root',
  'file:_DoubleMuon_Run2018D-12Nov2019_UL2018-v2_DQMIO/2334BC68-85EA-7D49-9ED3-E78A84FF1BEF.root',
)
process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('325172:1-325173:0')
