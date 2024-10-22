import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('skipEvents',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to skip")
options.register('debug',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "More verbose output")
options.parseArguments()

process = cms.Process('DRD')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring (options.inputFiles),
    skipEvents=cms.untracked.uint32(options.skipEvents)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(False),
    wantSummary = cms.untracked.bool(True),
)

# Output definition
process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *_*_*_DRD"),
    fileName = cms.untracked.string(options.outputFile),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('path'),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

# enable debug message logging for our modules
# process.MessageLogger = cms.Service(
#     "MessageLogger",
#     threshold  = cms.untracked.string('DEBUG'),
#     categories = cms.untracked.vstring('L1T'),
#     debugModules = cms.untracked.vstring(
#         'caloLayer1Digis',
#         'caloLayer1RawFed1354',
#         'caloLayer1RawFed1356',
#         'caloLayer1RawFed1358',
#     ),
#     destinations = cms.untracked.vstring('cerr'),
#     cerr = cms.untracked.PSet(
#         L1T = cms.untracked.PSet(
#             limit = cms.untracked.int32(-1),
#         ),
#         default = cms.untracked.PSet(
#             limit = cms.untracked.int32(0),
#         ),
#     ),
# )


# user stuff
process.load('EventFilter.L1TRawToDigi.caloLayer1Digis_cfi')
process.load('EventFilter.L1TRawToDigi.caloLayer1Raw_cfi')

for prod in [process.caloLayer1RawFed1354, process.caloLayer1RawFed1356, process.caloLayer1RawFed1358]:
    prod.ecalDigis = cms.InputTag("caloLayer1Digis")
    prod.hcalDigis = cms.InputTag("caloLayer1Digis")
    prod.caloRegions = cms.InputTag("caloLayer1Digis")

process.collectPackers = cms.EDProducer("RawDataCollectorByLabel",
    verbose = cms.untracked.int32(0),     # 0 = quiet, 1 = collection list, 2 = FED list
    RawCollectionList = cms.VInputTag(
        cms.InputTag('caloLayer1RawFed1354'),
        cms.InputTag('caloLayer1RawFed1356'),
        cms.InputTag('caloLayer1RawFed1358'),
    ),
)

# Path and EndPath definitions
process.path = cms.Path(
    process.caloLayer1Digis *
    process.collectPackers,
    process.caloLayer1Raw
)

process.out = cms.EndPath(
    process.output
)
