import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

options = VarParsing('analysis')
options.register ("unpack", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("l1", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("mc", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("dqm", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("useB904Data", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("run3", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("runCCLUT", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("runME11ILT", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("saveEdmOutput", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

process_era = Run3
if not options.run3:
      process_era = Run2_2018

process = cms.Process("L1CSCTPG", process_era)
process.load("Configuration/StandardSequences/GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.load("DQM.L1TMonitor.L1TdeCSCTPG_cfi")

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

## global tag (data or MC, Run-2 or Run-3)
from Configuration.AlCa.GlobalTag import GlobalTag
if options.mc:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
      if options.run3:
            process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')
else:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
      if options.run3:
            process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun3_Prompt_v5', '')

## running on unpacked data, or after running the unpacker
if not options.mc or options.unpack:
      process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
      process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"

## unpacker
if options.unpack:
      process.muonCSCDigis.DisableMappingCheck = options.useB904Data
      process.muonCSCDigis.B904Setup = options.useB904Data

## l1 emulator
l1csc = process.cscTriggerPrimitiveDigis
if options.l1:
      l1csc.commonParam.runCCLUT = options.runCCLUT
      l1csc.commonParam.runME11ILT = options.runME11ILT
      ## running on unpacked data, or after running the unpacker
      if (not options.mc or options.unpack):
            l1csc.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
            l1csc.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"
            ## GEM-CSC trigger enabled
            if options.runME11ILT:
                  l1csc.GEMPadDigiClusterProducer = "muonCSCDigis:MuonGEMPadDigiCluster"

## DQM monitor
if options.dqm:
      process.l1tdeCSCTPG.B904Setup = options.useB904Data
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
process.p2 = cms.Path(l1csc)
process.p3 = cms.Path(process.l1tdeCSCTPG)
process.p4 = cms.EndPath(process.DQMoutput)
process.p5 = cms.EndPath(process.output)
process.p6 = cms.EndPath(process.endOfProcess)

process.schedule = cms.Schedule()
## add the unpacker
if options.unpack:
      process.schedule.extend([process.p1])

## add the emulator
if options.l1:
      process.schedule.extend([process.p2])

## add DQM step 1
if options.dqm:
      process.schedule.extend([process.p3, process.p4])

if options.saveEdmOutput:
      process.schedule.extend([process.p5])

process.schedule.extend([process.p6])
