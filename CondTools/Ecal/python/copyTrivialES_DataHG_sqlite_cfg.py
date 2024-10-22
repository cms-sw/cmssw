import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.ESTrivialCondRetriever_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_31X_PRESHOWER'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB_DataHG.db'

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
    record = cms.string('ESTBWeightsRcd'),
    tag = cms.string('ESTBWeights_DataHG')
    ),

    cms.PSet(
    record = cms.string('ESPedestalsRcd'),
    tag = cms.string('ESPedestals_DataHG')
    ),
    
    cms.PSet(
    record = cms.string('ESChannelStatusRcd'),
    tag = cms.string('ESChannelStatus_DataHG')
    )
    )    
                                          )

process.dbCopy = cms.EDAnalyzer("ESDBCopy",
                                timetype = cms.string('runnumber'),
                                toCopy = cms.VPSet(

    cms.PSet(
    record = cms.string('ESTBWeightsRcd'),
    container = cms.string('ESTBWeights')
    ),

    cms.PSet(
    record = cms.string('ESPedestalsRcd'),
    container = cms.string('ESPedestals')
    ),
    
    cms.PSet(
    record = cms.string('ESChannelStatusRcd'),
    container = cms.string('ESChannelStatus')
    )
    )
                                )

process.p = cms.Path(process.dbCopy)

