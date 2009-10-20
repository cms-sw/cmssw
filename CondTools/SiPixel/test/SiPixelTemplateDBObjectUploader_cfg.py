import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("SiPixelTemplateDBUpload")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V10::All"

#MagFieldValue = 0
#MagFieldValue = 3.8
#MagFieldValue = 4

MagFieldValue = float(sys.argv[2])


tagversion = 'v0'
if(MagFieldValue==0):
    MagFieldString = '0'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0022.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0023.out")
    theDetIds      = cms.vuint32( 1, 2) # 0 is for all, 1 is Barrel, 2 is EndCap
    theTemplateIds = cms.vuint32(22,23)
elif(MagFieldValue==4):
    MagFieldString = '4'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0018.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0019.out")
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(18,19)
else:
    MagFieldString = '38'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0020.out",
        "CalibTracker/SiPixelESProducers/data/template_summary_zp0021.out")
#        "CalibTracker/SiPixelESProducers/data/template_summary_zp0024.out",
#        "CalibTracker/SiPixelESProducers/data/template_summary_zp0025.out",
#        "CalibTracker/SiPixelESProducers/data/template_summary_zp0026.out",
#        "CalibTracker/SiPixelESProducers/data/template_summary_zp0027.out")
#    theDetIds      = cms.vuint32(27,2,1,0)
#    theTemplateIds = cms.vuint32(12,8,6,4)
    theDetIds      = cms.vuint32( 1, 2)
    theTemplateIds = cms.vuint32(20,21)

template_base = 'SiPixelTemplateDBObject' + MagFieldString + 'T'
#theTemplateBaseString = cms.string(template_base)

print 'Uploading',template_base

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(0),
    authenticationPath = cms.untracked.string('.')
    ),
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:siPixelTemplates' + MagFieldString + 'T.db'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string(template_base + 'Rcd'),
    tag = cms.string(template_base + tagversion)
    ))
                                          )

process.uploader = cms.EDAnalyzer("SiPixelTemplateDBObjectUploader",
                                  siPixelTemplateCalibrations = files_to_upload,
                                  theTemplateBaseString = cms.string(template_base),
                                  Version = cms.double("2.0"),
                                  MagField = cms.double(MagFieldValue),
                                  detIds = theDetIds,
                                  templateIds = theTemplateIds
)


process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.uploader)
process.CondDBCommon.connect = 'sqlite_file:siPixelTemplates' + MagFieldString + 'T.db'
process.CondDBCommon.DBParameters.messageLevel = 0
process.CondDBCommon.DBParameters.authenticationPath = './'
