import FWCore.ParameterSet.Config as cms

# This cff file is based off of the L1T Phase-2 Menu group using
# the process name "REPR"


# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters using Emulator

from L1Trigger.L1CaloTrigger.l1tEGammaCrystalsEmulatorProducer_cfi import *
l1tEGammaClusterEmuProducer.ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","")


# --------------------------------------------------------------------------------------------
#
# ----    Produce the calibrated tower collection combining Barrel, HGCal, HF

from L1Trigger.L1CaloTrigger.l1tTowerCalibrationProducer_cfi import *
l1tTowerCalibrationProducer.L1HgcalTowersInputTag = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor","")
l1tTowerCalibrationProducer.l1CaloTowers = cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection","")



# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1CaloJets

from L1Trigger.L1CaloTrigger.l1tCaloJetProducer_cfi import *
l1tCaloJetProducer.l1CaloTowers = cms.InputTag("l1tTowerCalibrationProducer","L1CaloTowerCalibratedCollection","")
l1tCaloJetProducer.L1CrystalClustersInputTag = cms.InputTag("l1tEGammaClusterEmuProducer", "L1EGXtalClusterEmulator","")



# --------------------------------------------------------------------------------------------
#
# ----    Produce the CaloJet HTT Sums

from L1Trigger.L1CaloTrigger.l1tCaloJetHTTProducer_cfi import *



L1TCaloJetsSequence = cms.Sequence( 
        l1tEGammaClusterEmuProducer *
        l1tTowerCalibrationProducer *
        l1tCaloJetProducer *
        l1tCaloJetHTTProducer
)
