###############################################################################
# Way to use this:
#   cmsRun dumpHGCalTB_cfg.py type=TB230FEB
#
#   Options for type TB230FEB, TB230Jul
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "TB230FEB",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: TB230FEB, TB230Jul")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process("DumpHGCalT",Phase2C17I13M9)

geomFile = "Geometry.HGCalTBCommonData.test" + options.type + "XML_cfi"
outFile = "hgcal" + options.type + "DDD.root"
print("Geometry file: ", geomFile)
print("Output file: ", outFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()
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
