###############################################################################
# Way to use this:
#   cmsRun runHcalParametersFromDD4HepAnalyzer_cfg.py geometry=Run3
#
#   Options for geometry Run3, D105, D120, D121
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Run3",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: Run3, D105, D120, D121")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D105"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process("HcalParametersTest",Phase2C17I13M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtendedRun4D105Reco_cff')
elif (options.geometry == "D120"):
    from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process("HcalParametersTest",Phase2C26I13M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtendedRun4D120Reco_cff')
elif (options.geometry == "D121"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process("HcalParametersTest",Phase2C22I13M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtendedRun4D121Reco_cff')
else:
    from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
    process = cms.Process("HcalParametersTest",Run3_dd4hep)
    process.load("Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff")

process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.Geometry=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
