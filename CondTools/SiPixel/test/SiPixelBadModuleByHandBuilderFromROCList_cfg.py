import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as opts
process = cms.Process("ICALIB")

options = opts.VarParsing ('standard')

options.register('byLumi',
                 False,
                 opts.VarParsing.multiplicity.singleton,
                 opts.VarParsing.varType.bool,
                 'is output LS Based?')
options.register('outputFileName',
                 'SiPixelQualityTest',
                 opts.VarParsing.multiplicity.singleton,
                 opts.VarParsing.varType.string,
                 'output file name')
options.register('outputTagName',
                 'SiPixelQualityTest',
                 opts.VarParsing.multiplicity.singleton,
                 opts.VarParsing.varType.string,
                 'output tag name')
options.register('inputROCList',
                 'ROCList.txt',
                 opts.VarParsing.multiplicity.singleton,
                 opts.VarParsing.varType.string,
                 'input ROC to maks list')
options.parseArguments()

process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

# print("using EmptySource")
# process.source = cms.Source("EmptySource",
#                             numberEventsInRun = cms.untracked.uint32(1),
#                             firstRun = cms.untracked.uint32(1),
#                             numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#                             firstLuminosityBlock = cms.untracked.uint32(1))

print("using EmptyIOVSource")
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('lumiid' if options.byLumi else 'runnumber'),
                            firstValue = cms.uint64(4294967297 if options.byLumi else 1),
                            lastValue = cms.uint64(4294967297 if options.byLumi else 1),
                            interval = cms.uint64(1))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                          DBParameters = cms.PSet(
                                              authenticationPath = cms.untracked.string('')
                                          ),
                                          timetype = cms.untracked.string('lumiid' if options.byLumi else 'runnumber'),
                                          connect = cms.string(('sqlite_file:%s.db') % options.outputFileName),
                                          toPut = cms.VPSet(cms.PSet(
                                              record = cms.string('SiPixelQualityFromDbRcd'),
                                              tag = cms.string(options.outputTagName)
                                          )))

process.prod = cms.EDAnalyzer("SiPixelBadModuleByHandBuilder",
                              BadModuleList = cms.untracked.VPSet(),
                              Record = cms.string('SiPixelQualityFromDbRcd'),
                              SinceAppendMode = cms.bool(True),
                              IOVMode = cms.string('LumiBlock' if options.byLumi else 'Run'),
                              printDebug = cms.untracked.bool(True),
                              doStoreOnDB = cms.bool(True),
                              TimeFromEndRun = cms.untracked.bool(True),
                              ROCListFile = cms.untracked.string(options.inputROCList))

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)
