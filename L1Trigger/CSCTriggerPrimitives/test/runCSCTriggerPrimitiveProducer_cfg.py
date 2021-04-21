import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register ("unpack", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("analyzeData", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("analyzeEmul", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("mc", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("FastSim", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

process = cms.Process("CSCTPEmulator", Run3)
process.load("Configuration/StandardSequences/GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.load("L1Trigger.CSCTriggerPrimitives.CSCTriggerPrimitivesReader_cfi")
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(options.maxEvents)
)

process.options = cms.untracked.PSet(
      SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.source = cms.Source(
      "PoolSource",
      fileNames = cms.untracked.vstring(options.inputFiles)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("debug"),
    debug = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("DEBUG"),
        # threshold = cms.untracked.string("WARNING"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring("cscTriggerPrimitiveDigis",
        "lctreader")
)

## global tag
from Configuration.AlCa.GlobalTag import GlobalTag
if options.mc:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')
else:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')
      options.FastSim = False

## Process customization
analyze = options.analyzeData or options.analyzeEmul

if options.unpack:
      process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
      process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"

if options.analyzeEmul:
      process.cscTriggerPrimitivesReader.emulLctsIn = True

if options.analyzeData:
      process.cscTriggerPrimitivesReader.dataLctsIn = True

if options.FastSim:
      process.cscTriggerPrimitivesReader.CSCSimHitProducer = "MuonSimHits:MuonCSCHits"

# Output
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("lcts.root"),
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('TPEHists.root')
)

## schedule and path definition
process.p1 = cms.Path(process.muonCSCDigis)
process.p2 = cms.Path(process.cscTriggerPrimitiveDigis)
process.p3 = cms.Path(process.cscTriggerPrimitivesReader)
process.endjob_step = cms.EndPath(process.output * process.endOfProcess)

process.schedule = cms.Schedule()
if options.unpack:
      process.schedule.extend([process.p1])

## always add the trigger itself
process.schedule.extend([process.p2])

if options.analyze:
      process.schedule.extend([process.p3])

process.schedule.extend([process.endjob_step])
