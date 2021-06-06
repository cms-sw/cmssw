import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.TotLumi = cms.untracked.double(1000.0)
process.EcalTrivialConditionRetriever.InstLumi = cms.untracked.double(5.0e+34)
process.EcalTrivialConditionRetriever.intercalibConstantsFile = cms.untracked.string("EcalIntercalibConstants_2011_V3_Bon_start_mc.xml") 
process.EcalTrivialConditionRetriever.intercalibConstantsMCFile = cms.untracked.string("EcalIntercalibConstantsMC_digi_2011_V3_Bon_mc.xml") 



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
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_TL1000_IL5E34_mc')
        ),
                      cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRcd'),
            tag = cms.string('EcalLaserAPDPNRatios_TL1000_IL5E34_mc')
        ),
                      cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string('EcalIntercalibConstants_TL1000_IL5E34_mc')
        ) ,                      cms.PSet(
            record = cms.string('EcalIntercalibConstantsMCRcd'),
            tag = cms.string('EcalIntercalibConstantsMC_TL1000_IL5E34_mc')
        )
                      )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        container = cms.string('EcalPedestals')
    ),
                       cms.PSet(
            record = cms.string('EcalLaserAPDPNRatiosRcd'),
            container = cms.string('EcalLaserAPDPNRatios')
        ),
                       cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            container = cms.string('EcalIntercalibConstants')
        ),
                       cms.PSet(
            record = cms.string('EcalIntercalibConstantsMCRcd'),
            container = cms.string('EcalIntercalibConstantsMC')
        ))
)



process.p = cms.Path(process.dbCopy)

