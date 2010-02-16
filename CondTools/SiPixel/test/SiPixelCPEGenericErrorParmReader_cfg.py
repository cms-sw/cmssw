import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelCPEGenericErrorParmReaderTest")
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
    record = cms.string('SiPixelCPEGenericErrorParmRcd'),
    tag = cms.string('SiPixelCPEGenericErrorParm')
    )),
                                      DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(0),
    authenticationPath = cms.untracked.string('.')
    ),
                                      catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
                                      timetype = cms.string('runnumber'),
                                      connect = cms.string('sqlite_file:siPixelCPEGenericErrorParm.db')
                                      )

process.reader = cms.EDAnalyzer("SiPixelCPEGenericErrorParmReader")

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.reader)
