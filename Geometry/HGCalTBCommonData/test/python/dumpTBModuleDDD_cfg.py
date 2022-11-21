###############################################################################
# Way to use this:
#   cmsRun dumpTBModuleDDD_cfg.py type=HGCalTBModule
#
#   Options for type HGCalTBModule, HGCalTBModuleX, AHcalModuleAlgo
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "HGCalTBModule",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: HGCalTBModule, HGCalTBModuleX, AHcalModuleAlgo")

### get and parse the command line arguments
options.parseArguments()
print(options)

process = cms.Process("DumpTBModule")

####################################################################
# Use the options
geomFile = "Geometry.HGCalTBCommonData.test" + options.type + "XML_cfi"
outFile = "dump" + options.type + "DDD.root"

print("Geometry file: ", geomFile)
print("Output file:   ", outFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))

process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(outFile))

process.p = cms.Path(process.dump)
