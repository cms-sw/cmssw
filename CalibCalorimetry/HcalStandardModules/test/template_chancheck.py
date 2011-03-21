# Variables inclosed in "< >" come from outside of this file (see pedestalProducePayloads.csh)

import FWCore.ParameterSet.Config as cms
process = cms.Process('chancheck')
process.load('CondCore.DBCommon.CondDBSetup_cfi')
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source('EmptySource',
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(<run>)
)
process.hcal_db_producer = cms.ESProducer('HcalDbProducer',
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)
process.es_pool = cms.ESSource('PoolDBESSource',
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    appendToDataLabel = cms.string('reference'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('HcalElectronicsMapRcd'),
            tag = cms.string('HcalElectronicsMap_v7.03_hlt')
        )
    ),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_HCAL'),
    authenticationMethod = cms.untracked.uint32(0)
)

process.es_hardcode = cms.ESSource('HcalHardcodeCalibrations',
    toGet = cms.untracked.vstring('GainWidths', 'channelQuality', 'ZSThresholds', 'RespCorrs')
)

process.es_ascii = cms.ESSource('HcalTextCalibrations',
    input = cms.VPSet(
        cms.PSet(
            object = cms.string('Pedestals'),
            file = cms.FileInPath("CalibCalorimetry/HcalStandardModules/test/<filename>")
        )
    ),
    appendToDataLabel = cms.string('update')
)

process.es_ascii2 = cms.ESSource('HcalTextCalibrations',
    input = cms.VPSet(
        cms.PSet(
            object = cms.string('Pedestals'),
            file = cms.FileInPath("CalibCalorimetry/HcalStandardModules/test/dump.txt")
        )
    ),
    appendToDataLabel = cms.string('reference')
)

process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')

process.mergepeds = cms.EDAnalyzer('HcalPedestalsChannelsCheck',
    runNumber = cms.untracked.int32(<run>),
    epsilon = cms.untracked.double(<threshold>)
)

process.p = cms.Path(process.mergepeds)
