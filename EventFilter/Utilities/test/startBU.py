import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import math

#Instructions:
#     before running this script make sure that you have access to %fffBaseDir%/ramdisk and %fffBaseDir%/data
#     these parameters can also be changed in startFU.py and startBU.py scripts
#user = os.environ['USER']

options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
                  100, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('buBaseDir',
                  'ramdisk/', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fffBaseDir',
                  '.', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "FFF base directory")

options.register ('maxLS',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Max LS to generate (0 to disable limit)")

options.register ('eventsPerLS',
                  105,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Max LS to generate (0 to disable limit)")

options.register ('eventsPerFile',
                  20,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of events per raw file")

options.register ('fedMeanSize',
                  1024,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Mean size of generated (fake) FED raw payload")

options.register ('frdFileVersion',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Generate raw files with FRD file header with version 1 or separate JSON files with 0")



options.parseArguments()

#try to create 'ramdisk' directory
try:
    os.makedirs(options.fffBaseDir+"/"+options.buBaseDir)
except:pass


cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("FAKEBU")
if options.maxLS==0:
  process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
  )
else:
  process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(options.eventsPerLS * options.maxLS)
  )

process.options = cms.untracked.PSet(
)

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring( 'cout' ),
                                    cout = cms.untracked.PSet( FwkReport =
                                                               cms.untracked.PSet(reportEvery = cms.untracked.int32(1000),
                                                                                  optionalPSet = cms.untracked.bool(True),
                                                                                  limit = cms.untracked.int32(10000000)
                                                                                  ),
                                                               threshold = cms.untracked.string( "INFO" )
                                                               )
                                    )

process.source = cms.Source("EmptySource",
     firstRun= cms.untracked.uint32(options.runNumber),
     numberEventsInLuminosityBlock = cms.untracked.uint32(options.eventsPerLS),
     numberEventsInRun       = cms.untracked.uint32(0)
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fffBaseDir+"/"+options.buBaseDir),
    buBaseDir = cms.untracked.string(options.fffBaseDir+"/"+options.buBaseDir),
    directorIsBU = cms.untracked.bool(True),
    useFileBroker = cms.untracked.bool(False),
    fileBrokerHost = cms.untracked.string("")
)

#throttle when running with no limit
if options.maxLS==0:
    process.EvFBuildingThrottle = cms.Service("EvFBuildingThrottle",
        highWaterMark = cms.untracked.double(0.80),
        lowWaterMark = cms.untracked.double(0.75),
        sleepmSecs  = cms.untracked.uint32(500))

process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(0))

process.s = cms.EDProducer("DaqFakeReader",
                           meanSize = cms.untracked.uint32(options.fedMeanSize),
                           width = cms.untracked.uint32(int(math.ceil(options.fedMeanSize/2.))),
                           injectErrPpm = cms.untracked.uint32(0)
                           )

process.out = cms.OutputModule("RawStreamFileWriterForBU",
    ProductLabel = cms.untracked.string("s"),
    numEventsPerFile= cms.untracked.uint32(options.eventsPerFile),
    frdFileVersion=cms.untracked.uint32(options.frdFileVersion),
    frdVersion=cms.untracked.uint32(5)
    )

process.p = cms.Path(process.s+process.a)

process.ep = cms.EndPath(process.out)
