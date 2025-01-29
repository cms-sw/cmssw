import FWCore.ParameterSet.Config as cms
process = cms.Process("FIRST")

from FWCore.Services.modules import Tracer, ZombieKillerService, JobReportService
process.add_(Tracer(dumpEventSetupInfo = True))
process.add_(ZombieKillerService())
process.add_(JobReportService())

process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO.limit = 100

process.options = cms.untracked.PSet(
    #forceEventSetupCacheClearOnNewRun = cms.untracked.bool(True),
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.maxEvents.input = 30

from FWCore.Modules.modules import EmptySource
process.source = EmptySource(
    firstRun = 1,
    firstLuminosityBlock = 1,
    firstEvent = 1,
    numberEventsInLuminosityBlock = 4,
    numberEventsInRun = 10
)

from FWCore.Integration.modules import DoodadESSource, WhatsItESProducer, ThingWithMergeProducer
process.DoodadESSource = DoodadESSource(appendToDataLabel = 'abc', test2 = 'z')


# ---------------------------------------------------------------

copyProcess = cms.Process("COPY")
process.addSubProcess(cms.SubProcess(copyProcess))

# The following services are intended to test the isProcessWideService
# function which is defined in some services. These services
# should never be constructed and ignored, because
# The service from the top level process should be used.
# They intentionally have an illegal parameter to fail
# if they are ever constructed.
# Intentionally not using the more modern python syntax to avoid
# python catching the illegal parameter
copyProcess.MessageLogger = cms.Service("MessageLogger",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.CPU = cms.Service("CPU",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.InitRootHandlers = cms.Service("InitRootHandlers",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.ZombieKillerService = cms.Service("ZombieKillerService",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.JobReportService = cms.Service("JobReportService",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.SiteLocalConfigService = cms.Service("SiteLocalConfigService",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.AdaptorConfig = cms.Service("AdaptorConfig",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.ResourceInformationService = cms.Service("ResourceInformationService",
    intentionallyIllegalParameter = cms.bool(True)
)
copyProcess.CondorStatusService = cms.Service("CondorStatusService",
    intentionallyIllegalParameter = cms.bool(True)
)

copyProcess.DoodadESSource = DoodadESSource(appendToDataLabel = 'abc', test2 = 'zz')

# ---------------------------------------------------------------

prodProcess = cms.Process("PROD")
copyProcess.addSubProcess(cms.SubProcess(prodProcess))

prodProcess.DoodadESSource = DoodadESSource(appendToDataLabel = 'abcd', test2 = 'z')

prodProcess.thingWithMergeProducer = ThingWithMergeProducer()

prodProcess.p1 = cms.Path(prodProcess.thingWithMergeProducer)

from FWCore.Modules.modules import EventSetupRecordDataGetter
prodProcess.get = EventSetupRecordDataGetter(
    toGet = [cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::Doodad/abcd')
    ) ],
    verbose = True
)

prodProcess.noPut = ThingWithMergeProducer(noPut = True)

from FWCore.Framework.modules import IntProducer, TestMergeResults, RunLumiEventAnalyzer
prodProcess.putInt = IntProducer(ivalue = 6)
prodProcess.putInt2 = IntProducer(ivalue = 6)
prodProcess.putInt3 = IntProducer(ivalue = 6)

prodProcess.getInt = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
      cms.InputTag("putInt"),
  ),
  expectedSum = cms.untracked.int32(180),
  inputTagsNotFound = cms.untracked.VInputTag(
  )
)

prodProcess.path1 = cms.Path(prodProcess.get * prodProcess.putInt * prodProcess.putInt2 * prodProcess.putInt3 * prodProcess.getInt)



prodProcess.path2 = cms.Path(prodProcess.noPut)

# ---------------------------------------------------------------

copy2Process = cms.Process("COPY2")
prodProcess.addSubProcess(cms.SubProcess(copy2Process))

copy2Process.DoodadESSource = DoodadESSource(appendToDataLabel = 'abc', test2 = 'z')

# ---------------------------------------------------------------

prod2Process = cms.Process("PROD2")
copy2Process.addSubProcess(cms.SubProcess(prod2Process,
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_putInt_*_*"),
))
prod2Process.DoodadESSource = DoodadESSource(appendToDataLabel = 'abc', test2 = 'zz')

prod2Process.WhatsItESProducer = WhatsItESProducer()

prod2Process.thingWithMergeProducer = ThingWithMergeProducer()

# Reusing some code I used for testing merging, although in this
# context it has nothing to do with merging.
prod2Process.testmerge = TestMergeResults(
    expectedProcessHistoryInRuns = [
        'PROD',            # Run 1
        'PROD2',
        'PROD',            # Run 2
        'PROD2',
        'PROD',            # Run 3
        'PROD2'
    ]
)

prod2Process.dependsOnNoPut = ThingWithMergeProducer(labelsToGet = ['noPut'])

prod2Process.test = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
1,   0,   0,
1,   1,   0,
1,   1,   1,
1,   1,   2,
1,   1,   3,
1,   1,   4,
1,   1,   0,
1,   2,   0,
1,   2,   5,
1,   2,   6,
1,   2,   7,
1,   2,   8,
1,   2,   0,
1,   3,   0,
1,   3,   9,
1,   3,   10,
1,   3,   0,
1,   0,   0,
2,   0,   0,
2,   1,   0,
2,   1,   1,
2,   1,   2,
2,   1,   3,
2,   1,   4,
2,   1,   0,
2,   2,   0,
2,   2,   5,
2,   2,   6,
2,   2,   7,
2,   2,   8,
2,   2,   0,
2,   3,   0,
2,   3,   9,
2,   3,   10,
2,   3,   0,
2,   0,   0,
3,   0,   0,
3,   1,   0,
3,   1,   1,
3,   1,   2,
3,   1,   3,
3,   1,   4,
3,   1,   0,
3,   2,   0,
3,   2,   5,
3,   2,   6,
3,   2,   7,
3,   2,   8,
3,   2,   0,
3,   3,   0,
3,   3,   9,
3,   3,   10,
3,   3,   0,
3,   0,   0
]
)

prod2Process.get = EventSetupRecordDataGetter(
    toGet = [cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::Doodad/abc')
    ) ],
    verbose = True
)

prod2Process.getInt = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
  ),
  expectedSum = cms.untracked.int32(0),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("putInt"),
  )
)

from IOPool.Output.modules import PoolOutputModule
prod2Process.out = PoolOutputModule(fileName = 'testSubProcess.root')

prod2Process.path1 = cms.Path(prod2Process.thingWithMergeProducer)

prod2Process.path2 = cms.Path(prod2Process.test*prod2Process.testmerge)

prod2Process.path3 = cms.Path(prod2Process.get*prod2Process.getInt)

prod2Process.path4 = cms.Path(prod2Process.dependsOnNoPut)

prod2Process.endPath1 = cms.EndPath(prod2Process.out)
# ---------------------------------------------------------------

prod2ProcessAlt = cms.Process("PROD2ALT")
copy2Process.addSubProcess(cms.SubProcess(prod2ProcessAlt,
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_putInt_*_*"),
))
prod2ProcessAlt.DoodadESSource = DoodadESSource(appendToDataLabel = 'abc', test2 = 'zz')

prod2ProcessAlt.WhatsItESProducer = WhatsItESProducer()

prod2ProcessAlt.thingWithMergeProducer = ThingWithMergeProducer()

# Reusing some code I used for testing merging, although in this
# context it has nothing to do with merging.
prod2ProcessAlt.testmerge = TestMergeResults(
    expectedProcessHistoryInRuns = [
        'PROD',            # Run 1
        'PROD2ALT',
        'PROD',            # Run 2
        'PROD2ALT',
        'PROD',            # Run 3
        'PROD2ALT'
    ]
)

prod2ProcessAlt.dependsOnNoPut = ThingWithMergeProducer(labelsToGet = ['noPut'])

prod2ProcessAlt.test = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
1,   0,   0,
1,   1,   0,
1,   1,   1,
1,   1,   2,
1,   1,   3,
1,   1,   4,
1,   1,   0,
1,   2,   0,
1,   2,   5,
1,   2,   6,
1,   2,   7,
1,   2,   8,
1,   2,   0,
1,   3,   0,
1,   3,   9,
1,   3,   10,
1,   3,   0,
1,   0,   0,
2,   0,   0,
2,   1,   0,
2,   1,   1,
2,   1,   2,
2,   1,   3,
2,   1,   4,
2,   1,   0,
2,   2,   0,
2,   2,   5,
2,   2,   6,
2,   2,   7,
2,   2,   8,
2,   2,   0,
2,   3,   0,
2,   3,   9,
2,   3,   10,
2,   3,   0,
2,   0,   0,
3,   0,   0,
3,   1,   0,
3,   1,   1,
3,   1,   2,
3,   1,   3,
3,   1,   4,
3,   1,   0,
3,   2,   0,
3,   2,   5,
3,   2,   6,
3,   2,   7,
3,   2,   8,
3,   2,   0,
3,   3,   0,
3,   3,   9,
3,   3,   10,
3,   3,   0,
3,   0,   0
]
)

prod2ProcessAlt.get = EventSetupRecordDataGetter(
    toGet = [cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::Doodad/abc')
    ) ],
    verbose = True
)

prod2ProcessAlt.getInt = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
  ),
  expectedSum = cms.untracked.int32(0),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("putInt"),
  )
)

prod2ProcessAlt.out = PoolOutputModule(fileName = 'testSubProcessAlt.root')

prod2ProcessAlt.path1 = cms.Path(prod2ProcessAlt.thingWithMergeProducer)

prod2ProcessAlt.path2 = cms.Path(prod2ProcessAlt.test*prod2ProcessAlt.testmerge)

prod2ProcessAlt.path3 = cms.Path(prod2ProcessAlt.get*prod2ProcessAlt.getInt)

prod2ProcessAlt.path4 = cms.Path(prod2ProcessAlt.dependsOnNoPut)

prod2ProcessAlt.endPath1 = cms.EndPath(prod2ProcessAlt.out)
