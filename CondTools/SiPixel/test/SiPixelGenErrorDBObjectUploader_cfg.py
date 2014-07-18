import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("SiPixelGenErrorDBUpload")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_70_V4::All"
#process.GlobalTag.globaltag = "START71_V1::All"

MagFieldValue = float(sys.argv[2])

print '\nMagField = %f \n' % (MagFieldValue)
#version = 'v2'
version = sys.argv[3]

if ( MagFieldValue==0 ):
    MagFieldString = '0'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0022.out",
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0023.out")
    theDetIds      = cms.vuint32( 1, 2) # 0 is for all, 1 is Barrel, 2 is EndCap theGenErrorIds = cms.vuint32(22,23)
elif ( MagFieldValue==4 or MagFieldValue==40 ):
    MagFieldString = '4'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0018.out",
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0019.out")
    theDetIds      = cms.vuint32( 1, 2)
    theGenErrorIds = cms.vuint32(18,19)
elif ( MagFieldValue==3.8 or MagFieldValue==38 ):
    MagFieldString = '38'
    files_to_upload = cms.vstring(
#        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0020.out",
#        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0021.out")
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0030.out",
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0031.out")
    theDetIds      = cms.vuint32( 1, 2)
    theGenErrorIds = cms.vuint32(30,31)
elif ( MagFieldValue==2 or MagFieldValue==20 ):
    MagFieldString = '2'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0030.out",
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0031.out")
    theDetIds      = cms.vuint32( 1, 2)
    theGenErrorIds = cms.vuint32(30,31)
elif ( MagFieldValue==3 or MagFieldValue==30 ):
    MagFieldString = '3'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0032.out",
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0033.out")
    theDetIds      = cms.vuint32( 1, 2)
    theGenErrorIds = cms.vuint32(32,33)
elif( MagFieldValue==3.5 or MagFieldValue==35 ):
    MagFieldString = '35'
    files_to_upload = cms.vstring(
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0034.out",
        "CalibTracker/SiPixelESProducers/data/generror_summary_zp0035.out")
    theDetIds      = cms.vuint32( 1, 2)
    theGenErrorIds = cms.vuint32(34,35)



generror_base = 'SiPixelGenErrorDBObject' + MagFieldString + 'T'
#theGenErrorBaseString = cms.string(generic_base)

print '\nUploading %s%s with record SiPixelGenErrorDBObjectRcd in file siPixelGenErrors%sT.db\n' % (generror_base,version,MagFieldString)

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
                                          connect = cms.string('sqlite_file:siPixelGenErrors' + MagFieldString + 'T.db'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelGenErrorDBObjectRcd'),
    tag = cms.string(generror_base + version)
    ))
                                          )

process.uploader = cms.EDAnalyzer("SiPixelGenErrorDBObjectUploader",
                                  siPixelGenErrorCalibrations = files_to_upload,
                                  theGenErrorBaseString = cms.string(generror_base),
                                  Version = cms.double("3.0"),
                                  MagField = cms.double(MagFieldValue),
                                  detIds = theDetIds,
                                  generrorIds = theGenErrorIds
)


process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.uploader)
process.CondDBCommon.connect = 'sqlite_file:siPixelGenErrors' + MagFieldString + 'T.db'
process.CondDBCommon.DBParameters.messageLevel = 0
process.CondDBCommon.DBParameters.authenticationPath = './'
