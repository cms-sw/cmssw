import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("TestSiStripNoises")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "141X_dataRun3_HLT_v1",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  375440,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "run number")

options.parseArguments()


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )

process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

print("Using Global Tag:", process.GlobalTag.globaltag._value)

process.SiStripNoisesGetAllChecker = cms.EDAnalyzer("SiStripNoisesGetAllChecker")

process.p = cms.Path(process.SiStripNoisesGetAllChecker)
