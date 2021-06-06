from __future__ import print_function

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register ("pack", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("unpack", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("reconstruct", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("view", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("validate", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("mc", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("useB904Data", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

## process def
process = cms.Process("TEST", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration/StandardSequences/GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("EventFilter.CSCRawToDigi.cscPacker_cfi")
process.load("EventFilter.CSCRawToDigi.viewDigi_cfi")

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

## global tag
from Configuration.AlCa.GlobalTag import GlobalTag
if options.mc:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')
else:
      process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')

process.DQMStore = cms.Service("DQMStore")

# customize messagelogger
process.MessageLogger.cerr.enable = False
process.MessageLogger.debugModules = cms.untracked.vstring('muonCSCDigis')
## categories: 'CSCDCCUnpacker|CSCRawToDigi', 'StatusDigis', 'StatusDigi', 'CSCRawToDigi', 'CSCDCCUnpacker', 'EventInfo',
process.MessageLogger.CSCDDUEventData = dict()
process.MessageLogger.CSCRawToDigi = dict()
process.MessageLogger.badData = dict()
process.MessageLogger.cout = cms.untracked.PSet(
      enable = cms.untracked.bool(True),
      INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
      ),
      #TRACE = cms.untracked.PSet(limit = cms.untracked.int32(0) ),
      noTimeStamps = cms.untracked.bool(False),
      #FwkReport = cms.untracked.PSet(
      #    reportEvery = cms.untracked.int32(1),
      #    limit = cms.untracked.int32(10000000)
      #),
      #CSCRawToDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
      #StatusDigi = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
      #EventInfo = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
      default = cms.untracked.PSet(limit = cms.untracked.int32(10000000)),
      #Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0)),
      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
      #FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0)),
      #FwkSummary = cms.untracked.PSet(reportEvery = cms.untracked.int32(1), limit = cms.untracked.int32(10000000) ),
      threshold = cms.untracked.string('DEBUG')
)

## modules
process.cscValidation = cms.EDAnalyzer(
      "CSCValidation",
      rootFileName = cms.untracked.string('cscv_RAW.root'),
      isSimulation = cms.untracked.bool(False),
      writeTreeToFile = cms.untracked.bool(True),
      useDigis = cms.untracked.bool(True),
      detailedAnalysis = cms.untracked.bool(False),
      useTriggerFilter = cms.untracked.bool(False),
      useQualityFilter = cms.untracked.bool(False),
      makeStandalonePlots = cms.untracked.bool(False),
      makeTimeMonitorPlots = cms.untracked.bool(True),
      rawDataTag = cms.InputTag("rawDataCollector"),
      alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
      clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
      corrlctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
      stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
      wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
      compDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
      cscRecHitTag = cms.InputTag("csc2DRecHits"),
      cscSegTag = cms.InputTag("cscSegments"),
      saMuonTag = cms.InputTag("standAloneMuons"),
      l1aTag = cms.InputTag("gtDigis"),
      hltTag = cms.InputTag("TriggerResults::HLT"),
      makeHLTPlots = cms.untracked.bool(True),
      simHitTag = cms.InputTag("g4SimHits", "MuonCSCHits")
)

process.cscValidation.isSimulation = options.mc

process.analyzer = cms.EDAnalyzer("DigiAnalyzer")

## customizations
if options.mc:
      val = process.cscValidation
      val.alctDigiTag = cms.InputTag("simCscTriggerPimitiveDigis")
      val.clctDigiTag = cms.InputTag("simCscTriggerPimitiveDigis")
      val.corrlctDigiTag = cms.InputTag("simCscTriggerPimitiveDigis")
      val.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
      val.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
      val.compDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
else:
      pack = process.cscpacker
      pack.wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi")
      pack.stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi")
      pack.comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
      pack.alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi")
      pack.clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi")
      pack.preTriggerTag = cms.InputTag("simCscTriggerPrimitiveDigis")
      pack.correlatedLCTDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi")

if options.useB904Data:
      process.muonCSCDigis.DisableMappingCheck = True
      process.muonCSCDigis.B904Setup = True

process.out = cms.OutputModule(
      "PoolOutputModule",
      fileName = cms.untracked.string('output.root'),
      outputCommands = cms.untracked.vstring("keep *")
)

## schedule and path definition
process.p1 = cms.Path(process.cscpacker)
process.p2 = cms.Path(process.muonCSCDigis)
process.p3 = cms.Path(process.csc2DRecHits * process.cscSegments)
process.p4 = cms.Path(process.cscValidation)
process.p5 = cms.Path(process.viewDigi)
process.endjob_step = cms.EndPath(process.out * process.endOfProcess)

process.schedule = cms.Schedule()
if options.pack:
      process.schedule.extend([process.p1])
if options.unpack:
      process.schedule.extend([process.p2])
if options.reconstruct:
      process.schedule.extend([process.p3])
if options.validate:
      process.schedule.extend([process.p4])
if options.view:
      process.schedule.extend([process.p5])

process.schedule.extend([process.endjob_step])
