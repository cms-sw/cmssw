import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.TFileService = cms.Service("TFileService",

  fileName = cms.string("dcs.root")
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",

  process.CondDBSetup,

  connect = cms.string('sqlite_file:pixelPVSSCond.db'),
  timetype = cms.untracked.string('runnumber'),
  logconnect = cms.untracked.string('sqlite_file:log.db'),

  toGet = cms.VPSet
  (
    cms.PSet( record = cms.string('PixelCaenChannelRcd'), tag = cms.string('All') )
  )
)

process.CaenChannel = cms.EDAnalyzer("PixelDCSObjectReader<PixelCaenChannelRcd>")

process.p = cms.Path(process.CaenChannel)
