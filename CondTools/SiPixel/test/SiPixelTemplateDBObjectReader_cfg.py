import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelTemplateDBReaderTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

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
#process.GlobalTag.globaltag = 'IDEAL_30X::All'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBSetup,
                                      loadAll = cms.bool(True),
                                      toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelTemplateDBObjectRcd'),
    tag = cms.string('SiPixelTemplateDBObject')
    )),
                                      DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(0),
    authenticationPath = cms.untracked.string('.')
    ),
                                      catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
                                      timetype = cms.string('runnumber'),
                                      connect = cms.string('sqlite_file:siPixelTemplates.db')
                                      )

process.reader = cms.EDFilter("SiPixelTemplateDBObjectReader",
                              siPixelTemplateCalibrationLocation = cms.string(
                             "CalibTracker/SiPixelESProducers"),
                              wantDetailedTemplateDBErrorOutput = cms.bool(wantDetailedOutput),
                              wantFullTemplateDBOutput = cms.bool(wantFullOutput))

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.reader)






