###############################################################################
# Way to use this:
#   cmsRun protoHGCalSimWatcher_cfg.py geometry=2015
#
#   Options for geometry 2015, 2016, 2017, 2017Plan1, 2018
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("HcalParametersTest")

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2015, 2016, 2017,2017Plan1, 2018, ")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "2015"):
    fileName = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2015.xml"
elif (options.geometry == "2016"):
    fileName = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2016.xml"
elif (options.geometry == "2017"):
    fileName = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2017.xml"
elif (options.geometry == "2017Plan1"):
    fileName = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2017Plan1.xml"
elif (options.geometry == "2018"):
    fileName = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2018.xml"
else:
    fileName = "Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry.xml"

print("Geometry File: ", fileName)

process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath(fileName),
                                            appendToDataLabel = cms.string('')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                appendToDataLabel = cms.string('')
)

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")
process.hcalParameters.fromDD4hep = cms.bool(True)
process.hcalSimulationParameters.fromDD4hep = cms.bool(True)

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
