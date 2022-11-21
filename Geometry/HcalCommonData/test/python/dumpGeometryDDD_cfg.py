###############################################################################
# Way to use this:
#   cmsRun dumpGeometryDDD_cfg.py geometry=2021
#
#   Options for geometry 2015, 2016, 2017, 2018, 2021
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2021",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2015, 2016, 2017, 2018, 2021")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

process = cms.Process('GeomDump')

geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
fileName = "hcal" + options.geometry + "DDD.root"

print("Geometry file: ", geomFile)
print("Output file:   ", fileName)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HCalGeom=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))


process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(fileName)
)

process.p = cms.Path(process.dump)
