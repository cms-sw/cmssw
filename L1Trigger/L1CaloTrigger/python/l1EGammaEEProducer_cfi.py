import FWCore.ParameterSet.Config as cms

l1EGammaEEProducer = cms.EDProducer("L1EGammaEEProducer",
                                    calibrationConfig = cms.PSet(calirationFile = cms.FileInPath('L1Trigger/L1CaloTrigger/data/calib_ee_v1.json')),
                                    Multiclusters=cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster3D'))
