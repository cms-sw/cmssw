import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff")
process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.EcalTrigPrimESProducer.DatabaseFile = 'TPG_beamv5_MC_startup.txt.gz'

process.load("CondCore.DBCommon.CondDBCommon_cfi")

## to write in offline DB:
#process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_34X_ECAL')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_ECAL')

process.CondDBCommon.connect = cms.string('sqlite_file:DB_tpg.db')

process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')

## to write in sql file:
#process.CondDBCommon.connect = 'sqlite_file:DB_beamv5_startup_v4_mc.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    messagelevel = cms.untracked.uint32(3),
#    catalog = cms.untracked.string('file:PoolFileCatalog_DB.xml'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGPedestalsRcd'),
        tag = cms.string('EcalTPGPedestals_test_2011')
    ), 
        cms.PSet(
            record = cms.string('EcalTPGLinearizationConstRcd'),
            tag = cms.string('EcalTPGLinearizationConst_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGSlidingWindowRcd'),
            tag = cms.string('EcalTPGSlidingWindow_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
            tag = cms.string('EcalTPGFineGrainEBIdMap_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainStripEERcd'),
            tag = cms.string('EcalTPGFineGrainStripEE_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainTowerEERcd'),
            tag = cms.string('EcalTPGFineGrainTowerEE_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutIdMapRcd'),
            tag = cms.string('EcalTPGLutIdMap_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightIdMapRcd'),
            tag = cms.string('EcalTPGWeightIdMap_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightGroupRcd'),
            tag = cms.string('EcalTPGWeightGroup_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutGroupRcd'),
            tag = cms.string('EcalTPGLutGroup_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBGroupRcd'),
            tag = cms.string('EcalTPGFineGrainEBGroup_test_2011')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGPhysicsConstRcd'),
            tag = cms.string('EcalTPGPhysicsConst_test_2011')
        ),
	cms.PSet(
            record = cms.string('EcalTPGSpikeRcd'),
            tag = cms.string('EcalTPGSpike_test_2011')
        ),
        cms.PSet(
            record = cms.string('EcalTPGCrystalStatusRcd'),
            tag = cms.string('EcalTPGCrystalStatus_test_2011')
        ),
        cms.PSet(
            record = cms.string('EcalTPGTowerStatusRcd'),
            tag = cms.string('EcalTPGTowerStatus_test_2011')
        ),
        cms.PSet(
            record = cms.string('EcalTPGStripStatusRcd'),
            tag = cms.string('EcalTPGStripStatus_test_2011')
        ))
)

process.dbCopy = cms.EDAnalyzer("EcalTPGDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGPedestalsRcd'),
        container = cms.string('EcalTPGPedestals')
    ), 
        cms.PSet(
            record = cms.string('EcalTPGLinearizationConstRcd'),
            container = cms.string('EcalTPGLinearizationConst')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGSlidingWindowRcd'),
            container = cms.string('EcalTPGSlidingWindow')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
            container = cms.string('EcalTPGFineGrainEBIdMap')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainStripEERcd'),
            container = cms.string('EcalTPGFineGrainStripEE')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainTowerEERcd'),
            container = cms.string('EcalTPGFineGrainTowerEE')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutIdMapRcd'),
            container = cms.string('EcalTPGLutIdMap')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightIdMapRcd'),
            container = cms.string('EcalTPGWeightIdMap')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightGroupRcd'),
            container = cms.string('EcalTPGWeightGroup')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutGroupRcd'),
            container = cms.string('EcalTPGLutGroup')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBGroupRcd'),
            container = cms.string('EcalTPGFineGrainEBGroup')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGPhysicsConstRcd'),
            container = cms.string('EcalTPGPhysicsConst')
        ),
	cms.PSet(
            record = cms.string('EcalTPGSpikeRcd'),
            container = cms.string('EcalTPGSpike')
        ),
	cms.PSet(
            record = cms.string('EcalTPGCrystalStatusRcd'),
            container = cms.string('EcalTPGCrystalStatus')
        ),
	cms.PSet(
            record = cms.string('EcalTPGTowerStatusRcd'),
            container = cms.string('EcalTPGTowerStatus')
        ),
	cms.PSet(
            record = cms.string('EcalTPGStripStatusRcd'),
            container = cms.string('EcalTPGStripStatus')
        ))
)

process.p = cms.Path(process.dbCopy)



