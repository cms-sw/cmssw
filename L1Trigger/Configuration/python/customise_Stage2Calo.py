import FWCore.ParameterSet.Config as cms

def Stage2CaloFromRaw(process):

    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_cfi")

    process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
    process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("hcalDigis")

    process.stage2CaloPath = cms.Path(
        process.simCaloStage2Layer1Digis
        +process.simCaloStage2Digis
    )

    process.schedule.append(process.stage2CaloPath)

    return process

def Stage2CaloFromRaw_HWConfig(process):

    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi")
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_HWConfig_cfi")

    process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
    process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("hcalDigis")

    process.stage2CaloPath = cms.Path(
        process.simCaloStage2Layer1Digis
        +process.simCaloStage2Digis
    )

    process.schedule.append(process.stage2CaloPath)

    return process
