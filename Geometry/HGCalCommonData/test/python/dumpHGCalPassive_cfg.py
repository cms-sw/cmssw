###############################################################################
# Way to use this:
#   cmsRun dumpHGCalPassive_cfg.py geometry=V18 type=DDD
#
#   Options for geometry V18, V19, V19c
#               type     DDD, DD4hep
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "V18",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: V18, V19, V19c")
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")

### get and parse the command line arguments
options.parseArguments()

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('GeomDump',Phase2C17I13M9)
fileName = "hgcalPassive" + options.geometry + options.type + ".root"
if (options.type == "DD4hep"):
    geomFile = "Geometry/HGCalCommonData/data/dd4hep/testHGCalPassive" + options.geometry + ".xml"
    process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                                confGeomXMLFiles = cms.FileInPath(geomFile),
    appendToDataLabel = cms.string('DDHGCal'))
else:
    geomFile = "Geometry.HGCalCommonData.testHGCalPassive" + options.geometry + "XML_cfi"
    process.load(geomFile)

print("Geometry file: ", geomFile)
print("Output file:   ", fileName)

process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()
#   process.MessageLogger.SimG4CoreGeometry=dict()
#   process.MessageLogger.TGeoMgrFromDdd=dict()
    
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

if (options.type == "DD4hep"):
    process.dump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string(fileName),
                                  DDDetector = cms.ESInputTag('','DDHGCal'))
else:
    process.add_(cms.ESProducer("TGeoMgrFromDdd",
                                verbose = cms.untracked.bool(False),
                                level   = cms.untracked.int32(14)))
    process.dump = cms.EDAnalyzer("DumpSimGeometry",
                                  outputFileName = cms.untracked.string(fileName))

process.p = cms.Path(process.dump)
