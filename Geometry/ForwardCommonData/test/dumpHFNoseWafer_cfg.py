###############################################################################
# Way to use this:
#   cmsRun dumpHFNoseWafer_cfg.py type=V1
#
#   Options for type V1, V2
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "V2",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: V1, V2")

### get and parse the command line arguments
options.parseArguments()
print(options)

process = cms.Process("DUMP")

####################################################################
# Use the options
geomFile = "Geometry.ForwardCommonData.hfnoseWafer" + options.type + "XML_cfi"
outFile = "hfnoseWafer" + options.type + ".root"

print("Geometry file: ", geomFile)
print("Output file:   ", outFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()
#   process.MessageLogger.SimG4CoreApplication=dict()
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
                              outputFileName = cms.untracked.string(outFile))

process.p = cms.Path(process.dump)
