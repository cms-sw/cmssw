import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelTemplateDBReaderTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")


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

process.reader = cms.EDFilter("SiPixelTemplateDBObjectReader")

process.myprint = cms.OutputModule("AsciiOutputModule")

#process.Timing = cms.Service("Timing")

process.p = cms.Path(process.reader)
#process.ep = cms.EndPath(process.myprint)





