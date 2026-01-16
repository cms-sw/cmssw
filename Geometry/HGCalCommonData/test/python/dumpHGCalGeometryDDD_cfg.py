###############################################################################
# Way to use this:
#   cmsRun dumpHGCalGeometryDDD_cfg.py geometry=D121
#
#   Options for geometry D95, D96, D98, D99, D100, D101, D102, D103, D104,
#                        D105, D106, D107, D108, D109, D110, D111, D112, D113,
#                        D114, D115, D116, D117, D118, D119, D120, D121, D122,
#                        D123, D124, D125
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D121",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D95, D96, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115, D116, D117, D118, D119, D120, D121, D122, D123, D124, D125")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
geomName = "Run4" + options.geometry
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

process = cms.Process('GeomDump',ERA)
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
fileName = "CMS" + geomName + "DDD.root"

print("Geometry Name:   ", geomName)
print("Geometry file:   ", geomFile)
print("Output file:     ", fileName)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.add_(cms.ESProducer("TGeoMgrFromDdd",
                            verbose = cms.untracked.bool(False),
                            level = cms.untracked.int32(14)
                            ))

process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(fileName))

process.p = cms.Path(process.dump)
