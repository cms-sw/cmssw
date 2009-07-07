import FWCore.ParameterSet.Config as cms

process = cms.Process("peds")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("HcalTBSource",
    streams = cms.untracked.vstring('HCAL_Trigger', 'HCAL_DCC700', 'HCAL_DCC701', 'HCAL_DCC702', 'HCAL_DCC703', 'HCAL_DCC704', 'HCAL_DCC705', 
                                    'HCAL_DCC706', 'HCAL_DCC707', 'HCAL_DCC708', 'HCAL_DCC709', 'HCAL_DCC710', 'HCAL_DCC711', 'HCAL_DCC712', 
                                    'HCAL_DCC713', 'HCAL_DCC714', 'HCAL_DCC715', 'HCAL_DCC716', 'HCAL_DCC717', 'HCAL_DCC718', 'HCAL_DCC719', 
                                    'HCAL_DCC720', 'HCAL_DCC721', 'HCAL_DCC722', 'HCAL_DCC723', 'HCAL_DCC724', 'HCAL_DCC725', 'HCAL_DCC726', 
                                    'HCAL_DCC727', 'HCAL_DCC728', 'HCAL_DCC729', 'HCAL_DCC730', 'HCAL_DCC731'),
    fileNames = cms.untracked.vstring("file:/bigspool/usc/USC_101910.root")
)

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

process.es_pool = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('HcalQIEDataRcd'),
            tag = cms.string('HcalQIEData_NormalMode_v7.00_offline')
        ),
        cms.PSet(
            record = cms.string('HcalElectronicsMapRcd'),
            tag = cms.string('HcalElectronicsMap_v6.07_offline')
        )),
    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
    authenticationMethod = cms.untracked.uint32(0)
)

process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('Gains','GainWidths','Pedestals','PedestalWidths','channelQuality','ZSThresholds','RespCorrs','L1TriggerObjects','TimeCorrs','LUTCorrs','PFCorrs')
)


process.hcalDigis = cms.EDFilter("HcalRawToDigi",
    UnpackZDC = cms.untracked.bool(False),
    FilterDataQuality = cms.bool(True),
    ExceptionEmptyData = cms.untracked.bool(False),
    InputLabel = cms.InputTag("source"),
    ComplainEmptyData = cms.untracked.bool(False),
    UnpackCalib = cms.untracked.bool(True),
    lastSample = cms.int32(9),
    firstSample = cms.int32(0)
)

process.analyzepeds = cms.EDFilter("HcalCalibPeds")

process.p = cms.Path(process.hcalDigis*process.analyzepeds)
