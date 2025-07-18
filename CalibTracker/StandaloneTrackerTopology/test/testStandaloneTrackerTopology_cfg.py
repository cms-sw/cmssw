import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

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

options.register ('isPhase2',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "is phase2?")

options.parseArguments()

###################################################################
# Set default phase-2 settings
###################################################################
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

if(options.isPhase2):
    process = cms.Process("TopologyAnalysis",_PH2_ERA)
else:
    process = cms.Process("TopologyAnalysis")

###################################################################
# Message logger service
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

###################################################################
# Geometry producer and standard includes
###################################################################
process.load("Configuration.StandardSequences.Services_cff")

if(options.isPhase2):
    process.load("Configuration.Geometry.GeometryExtendedRun4Default_cff")
    process.load("Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff")
else:
    process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
if(options.isPhase2):
    process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')
else:
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

