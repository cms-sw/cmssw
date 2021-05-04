import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.laserAPDPNTime1 = cms.untracked.string('0')
process.EcalTrivialConditionRetriever.laserAPDPNTime2 = cms.untracked.string('1')
process.EcalTrivialConditionRetriever.laserAPDPNTime3 = cms.untracked.string('2')

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)


#    timetype = cms.untracked.string('timestamp'),

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('timestamp'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalLaserAlphasRcd'),
            tag = cms.string('EcalLaserAlphas_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRcd'),
            tag = cms.string('EcalLaserAPDPNRatios_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
            tag = cms.string('EcalLaserAPDPNRatiosRef_mc')
        ))
)


#    timetype = cms.string('timestamp'),

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
    timetype = cms.string('timestamp'),
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
        ))
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)

