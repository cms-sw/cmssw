import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
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

process.EcalTrigPrimESProducer.DatabaseFile = 'TPG_beamv7.txt.gz'

from CondCore.CondDB.CondDB_cfi import CondDB

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
     CondDB.clone(connect = cms.string("sqlite_file:TPG_beamv7.db")),
     timetype = cms.untracked.string("runnumber"),
     toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGPedestalsRcd'),
        tag = cms.string('EcalTPGPedestals_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGLinearizationConstRcd'),
            tag = cms.string('EcalTPGLinearizationConst_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGSlidingWindowRcd'),
            tag = cms.string('EcalTPGSlidingWindow_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
            tag = cms.string('EcalTPGFineGrainEBIdMap_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGFineGrainStripEERcd'),
            tag = cms.string('EcalTPGFineGrainStripEE_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGFineGrainTowerEERcd'),
            tag = cms.string('EcalTPGFineGrainTowerEE_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGLutIdMapRcd'),
            tag = cms.string('EcalTPGLutIdMap_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGWeightIdMapRcd'),
            tag = cms.string('EcalTPGWeightIdMap_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGWeightGroupRcd'),
            tag = cms.string('EcalTPGWeightGroup_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGOddWeightIdMapRcd'),
            tag = cms.string('EcalTPGOddWeightIdMap_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGOddWeightGroupRcd'),
            tag = cms.string('EcalTPGOddWeightGroup_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGTPModeRcd'),
            tag = cms.string('EcalTPGTPMode_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGLutGroupRcd'),
            tag = cms.string('EcalTPGLutGroup_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBGroupRcd'),
            tag = cms.string('EcalTPGFineGrainEBGroup_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGPhysicsConstRcd'),
            tag = cms.string('EcalTPGPhysicsConst_beamv7')
        ),
	cms.PSet(
            record = cms.string('EcalTPGSpikeRcd'),
            tag = cms.string('EcalTPGSpike_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGCrystalStatusRcd'),
            tag = cms.string('EcalTPGCrystalStatus_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGTowerStatusRcd'),
            tag = cms.string('EcalTPGTowerStatus_beamv7')
        ),
        cms.PSet(
            record = cms.string('EcalTPGStripStatusRcd'),
            tag = cms.string('EcalTPGStripStatus_beamv7')
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
            record = cms.string('EcalTPGOddWeightIdMapRcd'),
            container = cms.string('EcalTPGOddWeightIdMap')
        ),
        cms.PSet(
            record = cms.string('EcalTPGOddWeightGroupRcd'),
            container = cms.string('EcalTPGOddWeightGroup')
        ),
        cms.PSet(
            record = cms.string('EcalTPGTPModeRcd'),
            container = cms.string('EcalTPGTPMode')
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
