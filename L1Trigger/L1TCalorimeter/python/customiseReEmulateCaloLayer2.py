
import FWCore.ParameterSet.Config as cms


def reEmulateLayer2(process):

    process.load('L1Trigger/L1TCalorimeter/simCaloStage2Digis_cfi')

    process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis")

    process.caloLayer2 = cms.Path(
        process.simCaloStage2Digis
    )

    process.schedule.append(process.caloLayer2)

    return process
