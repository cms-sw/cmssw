import FWCore.ParameterSet.Config as cms

def Stage2CaloFromRaw(process):

    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_cfi")

    process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
    process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")

    # stuff for HF 1x1 TPs
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )
#    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/HcalCommonData/data/Phase0/hcalRecNumbering.xml')
#    process.XMLIdealGeometryESSource.geomXMLFiles.append('Geometry/HcalCommonData/data/Phase0/hcalRecNumberingRun2.xml')
    process.load("Geometry.HcalCommonData.testPhase0GeometryXML_cfi")

    process.es_pool = cms.ESSource(
        "PoolDBESSource",
        process.CondDBSetup,
        timetype = cms.string('runnumber'),
        toGet = cms.VPSet(
            cms.PSet(record = cms.string("HcalLutMetadataRcd"),
                     tag = cms.string("HcalLutMetadata_HFTP_1x1")
                     )
            ),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
        authenticationMethod = cms.untracked.uint32(0)
        )
    process.es_prefer_es_pool = cms.ESPrefer( "PoolDBESSource", "es_pool" )

    process.stage2CaloPath = cms.Path(
        process.simHcalTriggerPrimitiveDigis
        +process.simCaloStage2Layer1Digis
        +process.simCaloStage2Digis
    )

    process.schedule.append(process.stage2CaloPath)

    return process

def Stage2CaloFromRaw_HWConfig(process):

    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_HWConfig_cfi")

    process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
    process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis")

    # stuff for HF 1x1 TPs
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )
    process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/HcalCommonData/data/Phase0/hcalRecNumbering.xml')
    process.XMLIdealGeometryESSource.geomXMLFiles.append('Geometry/HcalCommonData/data/Phase0/hcalRecNumberingRun2.xml')

    process.es_pool = cms.ESSource(
        "PoolDBESSource",
        process.CondDBSetup,
        timetype = cms.string('runnumber'),
        toGet = cms.VPSet(
            cms.PSet(record = cms.string("HcalLutMetadataRcd"),
                     tag = cms.string("HcalLutMetadata_HFTP_1x1")
                     )
            ),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
        authenticationMethod = cms.untracked.uint32(0)
        )
    process.es_prefer_es_pool = cms.ESPrefer( "PoolDBESSource", "es_pool" )

    process.stage2CaloPath = cms.Path(
        process.simHcalTriggerPrimitiveDigis
        +process.simCaloStage2Layer1Digis
        +process.simCaloStage2Digis
    )

    process.schedule.append(process.stage2CaloPath)

    return process
