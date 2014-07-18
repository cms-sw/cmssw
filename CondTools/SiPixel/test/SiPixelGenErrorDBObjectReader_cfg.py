import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("SiPixelGenErrorDBReaderTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CalibTracker.SiPixelESProducers.SiPixelGenErrorDBObjectESProducer_cfi")

magfield = float(sys.argv[2])
#version = "v3"
version = sys.argv[3]

## Change to False if you do not want to test the global tag
testGlobalTag = False

if(magfield==0):
    magfieldString = "0T"
    magfieldCffStr = "0T"
elif(magfield==2   or magfield==20):
    magfieldString = "2T"
    magfieldCffStr = "20T"
elif(magfield==3   or magfield==30):
    magfieldString = "3T"
    magfieldCffStr = "30T"
elif(magfield==3.5 or magfield==35):
    magfieldString = "35T"
    magfieldCffStr = "35T"
elif(magfield==4   or magfield==40):
    magfieldString = "4T"
    magfieldCffStr = "40T"
else:
    magfieldString = "38T"
    magfieldCffStr = "38T"
    magfield = 3.8

#Load the correct Magnetic Field
process.load("Configuration.StandardSequences.MagneticField_"+magfieldCffStr+"_cff")

#Change to True if you would like a more detailed error output
wantDetailedOutput = False
#Change to True if you would like to output the full GenError database object
wantFullOutput = False

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

if testGlobalTag :
    process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#    process.GlobalTag.globaltag = "MC_70_V4::All"
    process.GlobalTag.globaltag = "START71_V1::All"
    
#Uncomment these two lines to get from the global tag
else:
    process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                          process.CondDBSetup,
                                          toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGenErrorDBObjectRcd'),
        tag = cms.string('SiPixelGenErrorDBObject' + magfieldString + version)
        )),
                                          timetype = cms.string('runnumber'),
                                          connect = cms.string('sqlite_file:siPixelGenErrors' + magfieldString + '.db')
                                          )
    process.PoolDBESSource.DBParameters.authenticationPath='.'
    process.PoolDBESSource.DBParameters.messageLevel=0

process.reader = cms.EDAnalyzer("SiPixelGenErrorDBObjectReader",
                              siPixelGenErrorCalibrationLocation = cms.string(
                             "CalibTracker/SiPixelESProducers"),
                              wantDetailedGenErrorDBErrorOutput = cms.bool(wantDetailedOutput),
                              wantFullGenErrorDBOutput = cms.bool(wantFullOutput),
                              TestGlobalTag = cms.bool(testGlobalTag)
                              )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.reader)






