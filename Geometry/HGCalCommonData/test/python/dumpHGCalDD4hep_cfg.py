###############################################################################
# Way to use this:
#   cmsRun dumpHGCalDD4hep_cfg.py geometry=D92
#
#   Options for geometry D77, D83, D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D77, D83, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9,dd4hep)
    geomFile = 'Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D83.xml'
    fileName = 'hgcalV15DD4hep.root'
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('DUMP',Phase2C11,dd4hep)
    geomFile = 'Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D77.xml'
    fileName = 'hgcalV14DD4hep.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9,dd4hep)
    geomFile = 'Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D92.xml'
    fileName = 'hgcalV17DD4hep.root'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9,Run3_dd4hep)
    geomFile = 'Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D93.xml'
    fileName = 'hgcalV17nDD4hep.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('DUMP',Phase2C11M9,dd4hep)
    geomFile = 'Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D88.xml'
    fileName = 'hgcalV16DD4hep.root'

print("Geometry file Name: ", geomFile)
print("Dump file Name: ", fileName)

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
