###############################################################################
# Way to use this:
#   cmsRun dumpGE0_cfg.py geometry=GE0
#
#   Options for geometry GE0, Mu24, GE21
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "GE0",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: GE0, Mu24, GE21")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
geomName = "Geometry.MuonCommonData.test" + options.geometry + "XML_cfi"
outFile = options.geometry + "DDD.root"

print("Geometry file: ", geomName)
print("Geometry file: ", outFile)

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process('Dump',Run3_DDD)

process.load(geomName)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.MuonGeom=dict()
#   process.MessageLogger.TGeoMgrFromDdd=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))


process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(outFile)
)

process.p = cms.Path(process.dump)
