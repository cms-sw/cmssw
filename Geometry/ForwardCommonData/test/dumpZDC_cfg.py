###############################################################################
# Way to use this:
#   cmsRun dumpZDC_cfg.py type=ZDCV2
#
#   Options for type ZDCV2, ZDC
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "ZDCV2",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: ZDCV2, ZDC")
### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
geomFile = "Geometry.ForwardCommonData.test" + options.type + "XML_cfi"
outFile = options.type + ".root"

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process("Dump",Run3_DDD)

print("Geom file Name:   ", geomFile)
print("Output file Name: ", outFile)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomFile)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
#   process.MessageLogger.SimG4CoreApplication=dict()
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
                              outputFileName = cms.untracked.string(outFile))

process.p = cms.Path(process.dump)
