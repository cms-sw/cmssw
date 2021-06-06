import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.ESTrivialCondRetriever_cfi")

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

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('ESADCToGeVConstantRcd'),
            tag = cms.string('ESADCToGeVConstant_mc')
        ),
        cms.PSet(
            record = cms.string('ESPedestalsRcd'),
            tag = cms.string('ESPedestals_mc')
        ),
        cms.PSet(
            record = cms.string('ESChannelStatusRcd'),
            tag = cms.string('EChannelStatus_mc')
        ),
        cms.PSet(
            record = cms.string('ESWeightStripGroupsRcd'),
            tag = cms.string('ESWeightStripGroups_mc')
        ),
        cms.PSet(
            record = cms.string('ESIntercalibConstantsRcd'),
            tag = cms.string('ESIntercalibConstants_mc')
        )) 
)

process.dbCopy = cms.EDAnalyzer("ESDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet(
        cms.PSet(
            record = cms.string('ESADCToGeVConstantRcd'),
            container = cms.string('ESADCToGeVConstant')
        ),
        cms.PSet(
            record = cms.string('ESPedestalsRcd'),
            container = cms.string('ESPedestals')
        ),
        cms.PSet(
            record = cms.string('ESChannelStatusRcd'),
            container = cms.string('ESChannelStatus')
        ),
        cms.PSet(
            record = cms.string('ESWeightStripGroupsRcd'),
            container = cms.string('ESWeightStripGroups')
        ),
        cms.PSet(
            record = cms.string('ESIntercalibConstantsRcd'),
            container = cms.string('ESIntercalibConstants')
        ))
)



process.p = cms.Path(process.dbCopy)

