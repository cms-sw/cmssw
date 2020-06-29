import FWCore.ParameterSet.Config as cms

l1EGammaEEProducer = cms.EDProducer("L1EGammaEEProducer",
                                    calibrationConfig = cms.PSet(calibrationFile = cms.FileInPath('L1Trigger/L1TCalorimeter/data/calib_ee_v1.json')),
                                    Multiclusters=cms.InputTag('hgcalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'))
