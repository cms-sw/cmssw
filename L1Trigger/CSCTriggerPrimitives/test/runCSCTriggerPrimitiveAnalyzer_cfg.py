import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register ("dataVsEmulation", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("analyzeEffiency", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("analyzeResolution", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("dataVsEmulationFile", "empty", VarParsing.multiplicity.singleton, VarParsing.varType.string)
"""
- For CMS runs, use the actual run number. Set B904Setup to False
- For B904 runs, set B904Setup to True and set runNumber >= 341761.
  Set B904RunNumber to when the data was taken, e.g. 210519_162820.
"""
options.register ("runNumber", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register ("B904Setup", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("B904RunNumber", "YYMMDD_HHMMSS", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

if options.B904Setup and options.B904RunNumber == "YYMMDD_HHMMSS":
    sys.exit("B904 setup was selected. Please provide a valid Run Number")

if (not options.B904Setup) and int(options.runNumber) == 0:
    sys.exit("Please provide a valid CMS Run Number")

process = cms.Process("ANALYSIS", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source(
      "PoolSource",
      fileNames = cms.untracked.vstring(options.inputFiles)
)

## if dataVsEmulation and analyzeEffiency or analyzeResolution are true,
## pick dataVsEmulation
if options.dataVsEmulation and (options.analyzeEffiency or options.analyzeResolution):
    options.analyzeEffiency = False
    options.analyzeResolution = False

if options.dataVsEmulation:
    options.maxEvents = 1
    process.source = cms.Source("EmptySource")


process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(options.maxEvents)
)

## customize the data vs emulator module
from DQM.L1TMonitor.L1TdeCSCTPG_cfi import l1tdeCSCTPGCommon
process.cscTriggerPrimitivesAnalyzer = cms.EDAnalyzer(
    "CSCTriggerPrimitivesAnalyzer",
    l1tdeCSCTPGCommon,
    ## file of the form "DQM_V0001_L1TEMU_R000334393"
    rootFileName = cms.string(options.dataVsEmulationFile),
    ## e.g. 334393
    runNumber = cms.uint32(options.runNumber),
    dataVsEmulatorPlots = cms.bool(options.dataVsEmulation),
    mcEfficiencyPlots = cms.bool(options.analyzeEffiency),
    mcResolutionPlots = cms.bool(options.analyzeResolution),
    B904RunNumber = cms.string(options.B904RunNumber)
)

# this needs to be set here, otherwise we duplicate the B904Setup parameter
process.cscTriggerPrimitivesAnalyzer.B904Setup = options.B904Setup

process.p = cms.Path(process.cscTriggerPrimitivesAnalyzer)
