###############################################################################
# Way to use this:
#   cmsRun dumpHGCalWaferDD4hep_cfg.py type=V17
#
#   Options for type V16, V17
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
                  "type of operations: V16, V17")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

process = cms.Process('GeomDump')

geomFile = "Geometry/HGCalCommonData/data/dd4hep/testHGCalWafer" + options.type + ".xml"
fileName = "hgcalWafer" + options.type + "DD4hep.root"

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
                                            appendToDataLabel = cms.string('')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string(fileName),
                                  DDDetector = cms.ESInputTag('','')
                                  )

process.p = cms.Path(process.testDump)
