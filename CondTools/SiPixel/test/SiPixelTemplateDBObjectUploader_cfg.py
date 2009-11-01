import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

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
                                          connect = cms.string('sqlite_file:siPixelTemplates.db'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelTemplateDBObjectRcd'),
    tag = cms.string('SiPixelTemplateDBObject')
    ))
                                          )

process.uploader = cms.EDAnalyzer("SiPixelTemplateDBObjectUploader",
                                  siPixelTemplateCalibrations = cms.vstring(
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0001.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0004.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0011.out",
    "CalibTracker/SiPixelESProducers/data/template_summary_zp0012.out"),
                                  Version = cms.double("1.3")
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.uploader)
process.CondDBCommon.connect = 'sqlite_file:siPixelTemplates.db'
process.CondDBCommon.DBParameters.messageLevel = 0
process.CondDBCommon.DBParameters.authenticationPath = './'
