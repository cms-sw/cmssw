import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import time

options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
                  100101, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('daqSourceMode',
                  '', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "DAQ source data mode")

options.register ('buBaseDir',
                  'ramdisk', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fuBaseDir',
                  'data', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fffBaseDir',
                  '.', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "FFF base directory")

options.register ('numThreads',
                  3, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  2, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW streams")

options.register ('cleanup',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "Delete any existing output directory")

options.parseArguments()

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("TESTFU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1) # ShmStreamConsumer requires synchronization at LuminosityBlock boundaries
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
    destinations = cms.untracked.vstring( 'cout' )
)

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1)
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(False),
    fileBrokerHostFromCfg = cms.untracked.bool(False),
    fileBrokerHost = cms.untracked.string("htcp40.cern.ch"),
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fffBaseDir+"/"+options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.fffBaseDir+"/"+options.buBaseDir),
    directorIsBU = cms.untracked.bool(False),
)

rundirectory = f"run{str(options.runNumber).zfill(6)}"

#look for input files
ram_dir_path=os.path.join(options.buBaseDir, rundirectory)
file_path_prefix = os.path.join(ram_dir_path, f"run{str(options.runNumber)}")
in_filelist = [
        file_path_prefix + "_ls0001_index000000.raw",
        file_path_prefix + "_ls0001_index000001.raw",
        file_path_prefix + "_ls0002_index000000.raw",
        file_path_prefix + "_ls0002_index000001.raw"
        ]
out_filelist = []
out_filelist_missing = []
for f in in_filelist:
    if os.path.exists(f):
        out_filelist.append(f)
    else:
        out_filelist_missing.append(f)

if not len(out_filelist):
    raise Exception(f"No input files from the list were found: {in_filelist}")
elif len(out_filelist) < len(in_filelist):
    print(f"Some files from the list were not found, running with available files. Missing: {out_filelist_missing}")
    time.sleep(1)

#check the state of output directory
workdir = os.path.join(options.fffBaseDir, options.fuBaseDir, rundirectory)

if os.path.exists(workdir):
    if options.cleanup:
        import shutil
        shutil.rmtree(workdir)
        os.makedirs(workdir)
    else:
        workdir_files = list(os.listdir(workdir))
        if len(workdir_files):
            print(f"WARNING: writeout directory {workfir} is not empty. Files will be mixed with the new run or overwritten.")
            print(f"Add cleanup=True to delete writeout directory it automatically")
            for f in workdir_files:
                print(f)
            time.sleep(2)
else:
    os.makedirs(workdir)


# Input source
process.source = cms.Source("DAQSource",
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True if options.daqSourceMode != "DTH" else False),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(2),
    eventChunkSize = cms.untracked.uint32(3),
    maxChunkSize = cms.untracked.uint32(100),
    numBuffers = cms.untracked.uint32(3),
    maxBufferedFiles = cms.untracked.uint32(2),
    fileListMode = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(out_filelist)
    )

process.PrescaleService = cms.Service( "PrescaleService",
                                       forceDefault = cms.bool( False ),
                                       prescaleTable = cms.VPSet( 
                                         cms.PSet(  pathName = cms.string( "p1" ),
                                         prescales = cms.vuint32( 10)
                                         ),
                                         cms.PSet(  pathName = cms.string( "p2" ),
                                         prescales = cms.vuint32( 100 )
                                         )
                                       ),
                                       lvl1DefaultLabel = cms.string( "Default" ),
                                       lvl1Labels = cms.vstring( 'Default' )
                                       )

process.filter1 = cms.EDFilter("HLTPrescaler",
                               L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
                               )
process.filter2 = cms.EDFilter("HLTPrescaler",
                               L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
                               )

sleepTime = 5
process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(sleepTime))

process.b = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(5))

process.tcdsRawToDigi = cms.EDProducer("TcdsRawToDigi",
    InputLabel = cms.InputTag("rawDataCollector")
)

if options.daqSourceMode == "DTH":
    process.p1 = cms.Path(process.a*process.filter1)
else:
    process.p1 = cms.Path(process.a*process.tcdsRawToDigi*process.filter1)

process.p2 = cms.Path(process.b*process.filter2)

process.streamA = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p1' ))
)

process.streamB = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)

process.streamC = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)

process.outRootFile = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p1', 'p2' )),
    fileName = cms.untracked.string('file:dth_output.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.hltJson = cms.EDAnalyzer("HLTriggerJSONMonitoring")

process.DQMStore = cms.Service( "DQMStore",
    verbose = cms.untracked.int32( 0 ),
    saveByLumi = cms.untracked.bool( False ),
)

from DQMServices.FileIO.DQMFileSaverPB_cfi import dqmSaver
process.hltDQMFileSaver = dqmSaver


process.daqHistoTest = cms.EDProducer("DaqTestHistograms",
    numberOfHistograms = cms.untracked.uint32(50),
    lumisectionRange =  cms.untracked.uint32(20)
)

process.ep = cms.EndPath(
  process.streamA
  + process.streamB
  + process.streamC
  + process.outRootFile
  + process.hltJson
  + process.daqHistoTest
  + process.hltDQMFileSaver
)
