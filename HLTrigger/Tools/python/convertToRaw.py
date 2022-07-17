# Convert the RAW data from EDM .root files into DAQ .raw format
#
# usage: cmsRun $CMSSW_RELEASE_BASE/HLTrigger/Tools/python/convertToRaw.py \
#           inputFiles=/store/path/file.root[,/store/path/file.root,...] \
#           runNumber=NNNNNN \
#           [lumiNumber=NNNN] \
#           [eventsPerFile=50] \
#           [eventsPerLumi=11650] \
#           [outputPath=output_directory]
#
# The output files will appear as output_directory/runNNNNNN/runNNNNNN_lumiNNNN_indexNNNNNN.raw .

import sys
import os
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("FAKE")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)                                 # to be overwritten after parsing the command line options
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()                             # to be overwritten after parsing the command line options
)

process.EvFDaqDirector = cms.Service( "EvFDaqDirector",
    runNumber = cms.untracked.uint32( 0 ),                          # to be overwritten after parsing the command line options
    baseDir = cms.untracked.string( "" ),                           # to be overwritten after parsing the command line options
    buBaseDir = cms.untracked.string( "" ),                         # to be overwritten after parsing the command line options
    useFileBroker = cms.untracked.bool( False ),
    fileBrokerKeepAlive = cms.untracked.bool( True ),
    fileBrokerPort = cms.untracked.string( "8080" ),
    fileBrokerUseLocalLock = cms.untracked.bool( True ),
    fuLockPollInterval = cms.untracked.uint32( 2000 ),
    requireTransfersPSet = cms.untracked.bool( False ),
    selectedTransferMode = cms.untracked.string( "" ),
    mergingPset = cms.untracked.string( "" ),
    outputAdler32Recheck = cms.untracked.bool( False ),
)

process.writer = cms.OutputModule("RawStreamFileWriterForBU",
    source = cms.InputTag('rawDataCollector'),
    numEventsPerFile = cms.uint32(0)                                # to be overwritten after parsing the command line options
)

process.endpath = cms.EndPath(process.writer)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 0                # to be overwritten after parsing the command line options

# parse command line options
options = VarParsing.VarParsing ('python')
for name in 'filePrepend', 'maxEvents', 'outputFile', 'secondaryOutputFile', 'section', 'tag', 'storePrepend', 'totalSections':
    del options._register[name]
    del options._beenSet[name]
    del options._info[name]
    del options._types[name]
    if name in options._singletons:
        del options._singletons[name]
    if name in options._lists:
        del options._lists[name]
    if name in options._noCommaSplit:
        del options._noCommaSplit[name]
    if name in options._noDefaultClear:
        del options._noDefaultClear[name]


options.register('runNumber',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number to use")

options.register('lumiNumber',
                 None,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Luminosity section number to use")

options.register('eventsPerLumi',
                 11650,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events in the given luminosity section to process")

options.register('eventsPerFile',
                 50,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Split the output into files with at most this number of events")

options.register('outputPath',
                 os.getcwd(),
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Output directory for the FED RAW data files")

options.parseArguments()

# check that the option values are valide
if options.runNumber <= 0:
    sys.stderr.write('Invalid run number\n')
    sys.exit(1)

if options.lumiNumber is not None and options.lumiNumber <= 0:
    sys.stderr.write('Invalid luminosity section number\n')
    sys.exit(1)

if options.eventsPerLumi == 0 or options.eventsPerLumi < -1:
    sys.stderr.write('Invalid number of events per luminosity section\n')
    sys.exit(1)

if options.eventsPerFile <= 0:
    sys.stderr.write('Invalid number of events per output file\n')
    sys.exit(1)

# configure the job based on the command line options
process.source.fileNames = options.inputFiles
if options.lumiNumber is not None:
    # process only one lumisection
    process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('%d:%d' % (options.runNumber, options.lumiNumber))
    process.maxEvents.input = options.eventsPerLumi
process.EvFDaqDirector.runNumber = options.runNumber
process.EvFDaqDirector.baseDir = options.outputPath
process.EvFDaqDirector.buBaseDir = options.outputPath
process.writer.numEventsPerFile = options.eventsPerFile
process.MessageLogger.cerr.FwkReport.reportEvery = options.eventsPerFile

# create the output directory, if it does not exist
outputRunPath = f'{options.outputPath}/run{options.runNumber:06d}'
os.makedirs(outputRunPath, exist_ok=True)
open(f'{outputRunPath}/fu.lock', 'w').close()
