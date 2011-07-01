import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.load("EcalTrivialAlpha_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_43X_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
#process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBCommon,
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAlphasRcd'),
      tag = cms.string('EcalLaserAlphas_v2_hlt')
    ), 
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRcd'),
      tag = cms.string('EcalLaserAPDPNRatios_v2_hlt')
    ), 
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
      tag = cms.string('EcalLaserAPDPNRatiosRef_v2_hlt')
    )
  )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
  timetype = cms.string('runnumber'),
  toCopy = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAlphasRcd'),
      container = cms.string('EcalLaserAlphas')
    ), 
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRcd'),
      container = cms.string('EcalLaserAPDPNRatios')
    ), 
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
      container = cms.string('EcalLaserAPDPNRatiosRef')
    )
  )
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)

