###############################################################################
# Way to use this:
#   cmsRun dumpHGCalGeometryDD4hep_cfg.py geometry=D110
#
#   Options for geometry D95, D96, D98, D99, D100, D101, D102, D103, D104, D105,
#                        D106, D107, D108, D109, D110, D111, D112, D113, D114,
#                        D115, D116
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D95, D96, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115, D116")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('GeomDump',Phase2C17I13M9)

geomFile = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometryRun4" + options.geometry + ".xml"
fileName = "CMSRun4" + options.geometry + "DD4hep.root"

print("Geometry file: ", geomFile)
print("Output file:   ", fileName)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath(geomFile),
                                            appendToDataLabel = cms.string('DDHGCal')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string(fileName),
                                  DDDetector = cms.ESInputTag('','DDHGCal')
                                  )

process.p = cms.Path(process.testDump)
