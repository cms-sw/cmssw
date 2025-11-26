###############################################################################
# Way to use this:
#   cmsRun testHGCalWaferSimWt_cfg.py geometry=D120
#
#   Options for geometry D95, D96, D98, D99, D100, D101, D102, D103, D104, 
#                        D105, D106, D107, D108, D109, D110, D111, D112, D113,
#                        D114, D115, D116, D117, D118, D119, D120, D121, D122,
#                        D123, D124, D125,
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D120",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D95, D96, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115, D116, D117, D118, D119, D120, D121, D122, D123, D124, D125")

### get and parse the command line arguments
options.parseArguments()

print(options)


####################################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('PROD',ERA)

process.load(geomFile)
process.load("Geometry.HGCalGeometry.hgcalWaferSimWt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeomX=dict()
#   process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.Timing = cms.Service("Timing")
process.hgcalWaferSimWt.debug = True

process.p1 = cms.Path(process.hgcalWaferSimWt)
