import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register ("dataVsEmulation", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register ("dataVsEmulationFile", "empty", VarParsing.multiplicity.singleton, VarParsing.varType.string)
"""
- For CMS runs, use the actual run number. Set useB904ME11, useB904ME21 or useB904ME234s2 to False
- For B904 runs, set useB904ME11, useB904ME21 or useB904ME234s2 to True and set runNumber >= 341761.
  Set B904RunNumber to when the data was taken, e.g. 210519_162820.
"""
options.register("runNumber", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("useB904ME11", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME1/1 data.")
options.register("useB904ME21", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME2/1 data (also works for ME3/1 and ME4/1).")
options.register("useB904ME234s2", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Set to True when using B904 ME1/1 data (also works for MEX/2 and ME1/3).")
options.register("B904RunNumber", "YYMMDD_HHMMSS", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

B904Setup = options.useB904ME11 or options.useB904ME21 or options.useB904ME234s2
if B904Setup and options.B904RunNumber == "YYMMDD_HHMMSS":
    sys.exit("B904 setup was selected. Please provide a valid Run Number")

if (not B904Setup) and int(options.runNumber) == 0:
    sys.exit("Please provide a valid CMS Run Number")

process = cms.Process("ANALYSIS", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source(
      "PoolSource",
      fileNames = cms.untracked.vstring(options.inputFiles)
)

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
    B904RunNumber = cms.string(options.B904RunNumber)
)

# this needs to be set here, otherwise we duplicate the B904Setup parameter
process.cscTriggerPrimitivesAnalyzer.useB904ME11 = options.useB904ME11
process.cscTriggerPrimitivesAnalyzer.useB904ME21 = options.useB904ME21
process.cscTriggerPrimitivesAnalyzer.useB904ME234s2 = options.useB904ME234s2

process.p = cms.Path(process.cscTriggerPrimitivesAnalyzer)
