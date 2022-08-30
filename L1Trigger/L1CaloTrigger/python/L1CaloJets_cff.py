import FWCore.ParameterSet.Config as cms

# This cff file is based off of the L1T Phase-2 Menu group using
# the process name "REPR"


# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters using Emulator

from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *
l1tEGammaClusterEmuProducer.ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","")


# --------------------------------------------------------------------------------------------
#
# ----    Produce the calibrated tower collection combining Barrel, HGCal, HF

from L1Trigger.L1CaloTrigger.L1TowerCalibrationProducer_cfi import *
L1TowerCalibrationProducer.L1HgcalTowersInputTag = cms.InputTag("hgcalTowerProducer","HGCalTowerProcessor","")
L1TowerCalibrationProducer.l1CaloTowers = cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection","")



# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1CaloJets

from L1Trigger.L1CaloTrigger.L1CaloJetProducer_cfi import *
L1CaloJetProducer.l1CaloTowers = cms.InputTag("L1TowerCalibrationProducer","L1CaloTowerCalibratedCollection","")
L1CaloJetProducer.L1CrystalClustersInputTag = cms.InputTag("l1tEGammaClusterEmuProducer", "L1EGXtalClusterEmulator","")



# --------------------------------------------------------------------------------------------
#
# ----    Produce the CaloJet HTT Sums

from L1Trigger.L1CaloTrigger.L1CaloJetHTTProducer_cfi import *



L1TCaloJetsSequence = cms.Sequence( 
        l1tEGammaClusterEmuProducer *
        l1tTowerCalibrationProducer *
        l1tCaloJetProducer *
        l1tCaloJetHTTProducer
)
