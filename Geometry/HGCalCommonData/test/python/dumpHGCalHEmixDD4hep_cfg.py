###############################################################################
# Way to use this:
#   cmsRun dumpHGCalHEmixDD4hep_cfg.py type=V17
#
#   Options for type V16, V17, V18
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
                  "type of operations: V16, V17, V18")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process("DumpHGCalHEmix",Phase2C17I13M9)

geomFile = "Geometry/HGCalCommonData/data/dd4hep/cms-test-ddhgcalHEmix" + options.type + "-algorithm.xml"
outFile = "hgcalHEmix" + options.type + "DD4hep.root"
print("Geometry file: ", geomFile)
print("Output file: ", outFile)

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

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath(geomFile),
                                            appendToDataLabel = cms.string('DDHGCalHEmix')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  outputFileName = cms.untracked.string(outFile),
                                  DDDetector = cms.ESInputTag('','DDHGCalHEmix')
                                  )

process.p = cms.Path(process.testDump)
