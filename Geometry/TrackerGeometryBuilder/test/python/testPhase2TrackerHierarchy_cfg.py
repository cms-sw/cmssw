import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing.VarParsing()
options.register('Scenario',
                 _settings.DEFAULT_VERSION, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "geometry version to use: Run4DXXX")
options.parseArguments()

###################################################################
# get Global Tag and ERA
###################################################################
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(options.Scenario)

process = cms.Process("GeometryTest",ERA)
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
if(options.Scenario == _settings.DEFAULT_VERSION):
    print("Loading default scenario: ", _settings.DEFAULT_VERSION)
    process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')
else:
    process.load('Configuration.Geometry.GeometryExtended'+options.Scenario+'Reco_cff')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.prod = cms.EDAnalyzer("GeoHierarchy",
    fromDDD = cms.bool(True)
)

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.p1 = cms.Path(process.prod)
