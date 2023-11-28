###############################################################################
# Way to use this:
#   cmsRun dumpHGCalDD4hep_cfg.py type=V17
#
#   Options for type V16, V17, V17Shift, V17n, V18
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
                  "geometry of operations: V16, V17, V17Shift, V17n, V18")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('DUMP',Phase2C17I13M9,dd4hep)

geomFile = "Geometry/HGCalCommonData/data/dd4hep/testHGCal" + options.type + ".xml"
outFile = "hgcal" + options.type + "DD4hep.root"

print("Geometry file Name: ", geomFile)
print("Dump file Name:     ", outFile)

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
                                  outputFileName = cms.untracked.string(outFile),
                                  DDDetector = cms.ESInputTag('','DDHGCal')
                                  )


process.p = cms.Path(process.testDump)
