import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register ("dataVsEmulation", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("analyzeEffiency", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("analyzeResolution", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("dataVsEmulationFile", "empty", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register ("runNumber", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.parseArguments()

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
)

process.p = cms.Path(process.cscTriggerPrimitivesAnalyzer)
