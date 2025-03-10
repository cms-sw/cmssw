###############################################################################
# Way to use this:
#   cmsRun dumpHGCalDDD_cfg.py type=V17
#
#   Options for type V16, V17, V17n, V17ng, V17Shift, V18, V18n, V18ng, V19
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS 
options = VarParsing.VarParsing('standard')
options.register('type',
                 "V17",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: V16, V17, V17n, V17ng, V17Shift, V18, V18n, V18ng, V19")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('GeomDump',Phase2C17I13M9)

geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + "XML_cfi"
fileName = "hgcal" + options.type + "DDD.root"

print("Geometry file: ", geomFile)
print("Output file:   ", fileName)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.TGeoMgrFromDdd=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))

process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(fileName))

process.p = cms.Path(process.dump)
