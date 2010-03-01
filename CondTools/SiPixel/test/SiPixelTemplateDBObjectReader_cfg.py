import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("SiPixelTemplateDBReaderTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

magfield = float(sys.argv[2])
#version = "v2"
version = sys.argv[3]

## Change to True if you want to test 3.8T standalone; others done by default;
testStandalone = False

if(magfield==0):
    magfieldString = "0T"
    testStandalone = True
elif(magfield==2):
    magfieldString = "2T"
    testStandalone = True
elif(magfield==3):
    magfieldString = "3T"
    testStandalone = True
elif(magfield==3.5 or magfield==35):
    magfieldString = "35T"
    testStandalone = True
elif(magfield==4):
    magfieldString = "4T"
    testStandalone = True
else:
    magfieldString = "38T"
    magfield = 3.8
#    testStandalone = True

#Change to True if you would like a more detailed error output
wantDetailedOutput = False
#Change to True if you would like to output the full template database object
wantFullOutput = False

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

#Uncomment these two lines to get from the global tag
#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = "MC_3XY_V23::All"

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBSetup,
                                      toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelTemplateDBObjectRcd'),
    tag = cms.string('SiPixelTemplateDBObject' + magfieldString + version)
    )),
                                      timetype = cms.string('runnumber'),
                                      connect = cms.string('sqlite_file:siPixelTemplates' + magfieldString + '.db')
                                      )
process.PoolDBESSource.DBParameters.authenticationPath='.'
process.PoolDBESSource.DBParameters.messageLevel=0

process.reader = cms.EDAnalyzer("SiPixelTemplateDBObjectReader",
                              siPixelTemplateCalibrationLocation = cms.string(
                             "CalibTracker/SiPixelESProducers"),
                              wantDetailedTemplateDBErrorOutput = cms.bool(wantDetailedOutput),
                              wantFullTemplateDBOutput = cms.bool(wantFullOutput),
                              MagneticField = cms.double(magfield),
                              TestStandalone = cms.bool(testStandalone)
                              )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.reader)






