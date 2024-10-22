import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("TopologyAnalysis")
options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:run2_data",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "run number")

options.parseArguments()

###################################################################
# Message logger service
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

###################################################################
# Geometry producer and standard includes
###################################################################
process.load("Configuration.StandardSequences.Services_cff")

if 'phase2' in options.globalTag:
    process.load("Configuration.Geometry.GeometryExtended2026D98_cff")
    process.load("Configuration.Geometry.GeometryExtended2026D98Reco_cff")
else:
    process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

###################################################################
# Empty Source
###################################################################
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

###################################################################
# The analysis module
###################################################################
process.myanalysis = cms.EDAnalyzer("StandaloneTrackerTopologyTest")

###################################################################
# Path
###################################################################
process.p1 = cms.Path(process.myanalysis)

