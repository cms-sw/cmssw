import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("reader")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

options = VarParsing ('analysis')
options.register("firstRun", 341761, VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "The first run for this data. Typically for Run-3 it needs to be set high enough, so that \
                  cmsRun recognizes GEM detectors as valid. That is the case starting run 341761. Beyond making \
                  sure that the GEM detectors are there, firstRun does not have a meaning.")
options.register("inputFilesGEM", "", VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "The GEM input file (if applicable). This needs to be filled for joint GEM-CSC runs with B904 ME1/1 test-stand.")
options.register("readGEMData", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when you want to process the GEM data as well as the CSC data.")
options.register("useB904ME11", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME1/1 data.")
options.register("useB904ME21", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME2/1 data (also works for ME3/1 and ME4/1).")
options.register("useB904ME234s2", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME1/1 data (also works for MEX/2 and ME1/3).")
options.maxEvents = 10000
options.parseArguments()

# Better to know actual number of events in the .raw data file to set maxEvents.
# Otherwise it doesn't stop automatically at the end of reading of .raw data file
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

## when reading GEM data and CSC data, use the FRDStreamSource
## this source can read the GEM data
if options.readGEMData:
    process.source = cms.Source(
        "FRDStreamSource",
        fileNames = cms.untracked.vstring(options.inputFilesGEM),
        verifyAdler32 = cms.untracked.bool(False),
        verifyChecksum = cms.untracked.bool(False),
        useL1EventID = cms.untracked.bool(False),
        firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID(*[cms.LuminosityBlockID(options.firstRun,0)]),
    )
## otherwise, just use an empty source
## reading the CSC data is done with a the CSCFileReader producer
else:
    process.source = cms.Source(
        "EmptySource",
        firstRun= cms.untracked.uint32(options.firstRun),
        numberEventsInLuminosityBlock = cms.untracked.uint32(200),
        numberEventsInRun       = cms.untracked.uint32(0)
    )

## this block ensures that the GEM output collection is named rawDataCollectorGEM, not source
process.rawDataCollectorGEM = cms.EDAlias(
    source = cms.VPSet(
        cms.PSet(
            type = cms.string('FEDRawDataCollection')
        )
    )
)

# For B904 setup ME11 chamber, which corresponds to ME+1/1/02 in the production system mapping
# changing to FED837 and RUI16 could let to pass data without disabling mapping consistency check unpacking flags
if options.useB904ME11:
    FEDRUI = cms.PSet(
        FED846 = cms.untracked.vstring('RUI01'),
        RUI01 = cms.untracked.vstring(options.inputFiles[0])
    )
# For B904 setup ME21 chamber, which corresponds to ME+2/1/03 VMECrate13 / DMBSlot2 RUI17 / FED838 in the production system mapping
elif options.useB904ME21:
    FEDRUI = cms.PSet(
        FED838 = cms.untracked.vstring('RUI17'),
        RUI17 = cms.untracked.vstring(options.inputFiles[0])
    )
# Please note that after passing mapping check this chamber still would be recognized as production chamber
# ME+2/2/03, which is OK, because this is the same chamber type as ME42 hardware-wise.
elif options.useB904ME234s2:
    FEDRUI = cms.PSet(
        FED839 = cms.untracked.vstring('RUI18'),
        RUI18 = cms.untracked.vstring(options.inputFiles[0])
    )
## default case
else:
    FEDRUI = cms.PSet(
        FED837 = cms.untracked.vstring('RUI16'),
        RUI16 = cms.untracked.vstring('/afs/cern.ch/user/b/barvic/public/cscgem_tests/csc_00000001_EmuRUI01_Local_000_210519_162820_UTC.raw')
    )

# Note by Sven Dildick: I had to change the CSC producer module name to process.rawDataCollectorCSC so
# that the name would not conflict with the GEM source.
process.rawDataCollectorCSC = cms.EDProducer(
    'CSCFileReader',
    FEDRUI,
    firstEvent  = cms.untracked.int32(0),
)

process.FEVT = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string(options.inputFiles[0].replace('.raw','.root')),
    outputCommands = cms.untracked.vstring(
        "keep *",
        ## drop the collection named "source" - incompatible with later CMSSW
        "drop FEDRawDataCollection_source_*_*",
        "keep FEDRawDataCollection_rawDataCollectorCSC_*_*",
        "keep FEDRawDataCollection_rawDataCollectorGEM_*_*"
    )
)

process.p = cms.Path(process.rawDataCollectorCSC)

process.outpath = cms.EndPath(process.FEVT)
