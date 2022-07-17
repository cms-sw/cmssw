import FWCore.ParameterSet.Config as cms
process = cms.Process("FIRST")

process.Tracer = cms.Service('Tracer',
  dumpEventSetupInfo = cms.untracked.bool(True)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO.limit = 100

process.options = cms.untracked.PSet(
    #forceEventSetupCacheClearOnNewRun = cms.untracked.bool(True),
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(4),
    numberEventsInRun = cms.untracked.uint32(10)
)

process.DoodadESSource = cms.ESSource("DoodadESSource"
                                      , appendToDataLabel = cms.string('abc')
                                      , test2 = cms.untracked.string('z')
)

process.CPU = cms.Service("CPU")
process.ZombieKillerService = cms.Service("ZombieKillerService")
process.JobReportService = cms.Service("JobReportService")

# ---------------------------------------------------------------

copyProcess = cms.Process("COPY")
process.addSubProcess(cms.SubProcess(copyProcess))

# The following services are intended to test the isProcessWideService
# function which is defined in some services. These services
# should never be constructed and ignored, because
# The service from the top level process should be used.
# They intentionally have an illegal parameter to fail
# if they are ever constructed.
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

copyProcess.DoodadESSource = cms.ESSource("DoodadESSource"
                                          , appendToDataLabel = cms.string('abc')
                                          , test2 = cms.untracked.string('zz')
)

# ---------------------------------------------------------------

prodProcess = cms.Process("PROD")
copyProcess.addSubProcess(cms.SubProcess(prodProcess))

prodProcess.DoodadESSource = cms.ESSource("DoodadESSource"
                                          , appendToDataLabel = cms.string('abcd')
                                          , test2 = cms.untracked.string('z')
)

prodProcess.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

prodProcess.p1 = cms.Path(prodProcess.thingWithMergeProducer)

prodProcess.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::Doodad/abcd')
    ) ),
    verbose = cms.untracked.bool(True)
)

prodProcess.noPut = cms.EDProducer("ThingWithMergeProducer",
    noPut = cms.untracked.bool(True)
)

prodProcess.putInt = cms.EDProducer("IntProducer",
    ivalue = cms.int32(6)
)

prodProcess.putInt2 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(6)
)

prodProcess.putInt3 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(6)
)

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

copy2Process.DoodadESSource = cms.ESSource("DoodadESSource"
                                           , appendToDataLabel = cms.string('abc')
                                           , test2 = cms.untracked.string('z')
)

# ---------------------------------------------------------------

prod2Process = cms.Process("PROD2")
copy2Process.addSubProcess(cms.SubProcess(prod2Process,
    outputCommands = cms.untracked.vstring(
        "keep *", 
        "drop *_putInt_*_*"),
))
prod2Process.DoodadESSource = cms.ESSource("DoodadESSource"
                                           , appendToDataLabel = cms.string('abc')
                                           , test2 = cms.untracked.string('zz')
)

prod2Process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

prod2Process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

# Reusing some code I used for testing merging, although in this
# context it has nothing to do with merging.
prod2Process.testmerge = cms.EDAnalyzer("TestMergeResults",

    expectedProcessHistoryInRuns = cms.untracked.vstring(
        'PROD',            # Run 1
        'PROD2',
        'PROD',            # Run 2
        'PROD2',
        'PROD',            # Run 3
        'PROD2'
    )
)

prod2Process.dependsOnNoPut = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('noPut')
)

prod2Process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
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
)
)

prod2Process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::Doodad/abc')
    ) ),
    verbose = cms.untracked.bool(True)
)

prod2Process.getInt = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
  ),
  expectedSum = cms.untracked.int32(0),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("putInt"),
  )
)

prod2Process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSubProcess.root')
)

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
prod2ProcessAlt.DoodadESSource = cms.ESSource("DoodadESSource"
                                           , appendToDataLabel = cms.string('abc')
                                           , test2 = cms.untracked.string('zz')
)

prod2ProcessAlt.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

prod2ProcessAlt.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

# Reusing some code I used for testing merging, although in this
# context it has nothing to do with merging.
prod2ProcessAlt.testmerge = cms.EDAnalyzer("TestMergeResults",

    expectedProcessHistoryInRuns = cms.untracked.vstring(
        'PROD',            # Run 1
        'PROD2ALT',
        'PROD',            # Run 2
        'PROD2ALT',
        'PROD',            # Run 3
        'PROD2ALT'
    )
)

prod2ProcessAlt.dependsOnNoPut = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('noPut')
)

prod2ProcessAlt.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
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
)
)

prod2ProcessAlt.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GadgetRcd'),
        data = cms.vstring('edmtest::Doodad/abc')
    ) ),
    verbose = cms.untracked.bool(True)
)

prod2ProcessAlt.getInt = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
  ),
  expectedSum = cms.untracked.int32(0),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("putInt"),
  )
)

prod2ProcessAlt.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSubProcessAlt.root')
)

prod2ProcessAlt.path1 = cms.Path(prod2ProcessAlt.thingWithMergeProducer)

prod2ProcessAlt.path2 = cms.Path(prod2ProcessAlt.test*prod2ProcessAlt.testmerge)

prod2ProcessAlt.path3 = cms.Path(prod2ProcessAlt.get*prod2ProcessAlt.getInt)

prod2ProcessAlt.path4 = cms.Path(prod2ProcessAlt.dependsOnNoPut)

prod2ProcessAlt.endPath1 = cms.EndPath(prod2ProcessAlt.out)
