import FWCore.ParameterSet.Config as cms

l1EGammaEEProducer = cms.EDProducer("L1EGammaEEProducer",
    Multiclusters = cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    calibrationConfig = cms.PSet(
        calibrationFile = cms.FileInPath('L1Trigger/L1TCalorimeter/data/calib_ee_v1.json')
    )
)

