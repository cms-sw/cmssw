import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

options = VarParsing('analysis')
options.register ("unpack", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("mc", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("dqm", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

process = cms.Process("CSCTPEmulator", Run2_2018)
process.load("Configuration/StandardSequences/GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.load("DQM.L1TMonitor.L1TdeCSCTPG_cfi")
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')

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
          threshold = cms.untracked.string("WARNING"),
          lineLength = cms.untracked.int32(132),
          noLineBreaks = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring("cscTriggerPrimitiveDigis")
)

## global tag
from Configuration.AlCa.GlobalTag import GlobalTag
if options.mc:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')
else:
      #process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun3_Prompt_v5', '')
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

## running on unpacked data, or after running the unpacker
if not options.mc or options.unpack:
      process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
      process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"

## DQM monitor
if options.dqm:
      process.l1tdeCSCTPG.emulALCT = "cscTriggerPrimitiveDigis"
      process.l1tdeCSCTPG.emulCLCT = "cscTriggerPrimitiveDigis"
      process.l1tdeCSCTPG.emulLCT = "cscTriggerPrimitiveDigis:MPCSORTED"


# Output
process.output = cms.OutputModule(
    "PoolOutputModule",
      outputCommands = cms.untracked.vstring(
            ['keep *',
             'drop *_rawDataCollector_*_*',
      ]),
      fileName = cms.untracked.string("lcts2.root"),
)

## DQM output
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step_DQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

## schedule and path definition
process.p1 = cms.Path(process.muonCSCDigis)
process.p2 = cms.Path(process.cscTriggerPrimitiveDigis)
process.p3 = cms.EndPath(process.DQMoutput)
process.endjob_step = cms.EndPath(process.output * process.endOfProcess)

process.schedule = cms.Schedule()
if options.unpack:
      process.schedule.extend([process.p1])

## always add the trigger itself
process.schedule.extend([process.p2])

## add DQM step 1
if options.dqm:
      process.schedule.extend([process.p3])

process.schedule.extend([process.endjob_step])
