import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelCPEGenericErrorParmReaderTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

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

process.reader = cms.EDFilter("SiPixelCPEGenericErrorParmReader")
#process.reader = cms.EDFilter("PxCPEdbReader")

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.reader)
