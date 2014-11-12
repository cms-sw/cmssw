import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloStage1Params_cfi import *

# HCAL TP hack
from L1Trigger.L1TCalorimeter.L1TRerunHCALTP_FromRaw_cff import *

### CCLA include latest RCT calibrations from UCT
from L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff import *

from Configuration.StandardSequences.RawToDigi_Data_cff import ecalDigis

# RCT
# HCAL input would be from hcalDigis if hack not needed
from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )

# stage 1 itself
from L1Trigger.L1TCalorimeter.L1TCaloStage1_cff import *

#
# Update HfRing thresholds to accomodate di-iso tau trigger thresholds
from L1TriggerConfig.L1ScalesProducers.l1CaloScales_cfi import l1CaloScales
l1CaloScales.L1HfRingThresholds = cms.vdouble(0.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0)

l1CaloScales.L1HtMissThresholds = cms.vdouble(
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
    0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
    0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
    0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,
    0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60,
    0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70,
    0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80,
    0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90,
    0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01,
    1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01,
    1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01,
    1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01
    )

## l1CaloScales.L1HtMissThresholds = cms.vdouble(
##      0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
##     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
##     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
##     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
##     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
##     51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
##     61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
##     71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
##     81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
##     91, 92, 93, 94, 95, 96, 97, 98, 99, 100
## )

# L1Extra
import L1Trigger.Configuration.L1Extra_cff
l1ExtraLayer2 = L1Trigger.Configuration.L1Extra_cff.l1extraParticles.clone()
l1ExtraLayer2.isolatedEmSource    = cms.InputTag("simCaloStage1LegacyFormatDigis","isoEm")
l1ExtraLayer2.nonIsolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","nonIsoEm")

l1ExtraLayer2.forwardJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","forJets")
l1ExtraLayer2.centralJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","cenJets")
l1ExtraLayer2.tauJetSource     = cms.InputTag("simCaloStage1LegacyFormatDigis","tauJets")
l1ExtraLayer2.isoTauJetSource  = cms.InputTag("simCaloStage1LegacyFormatDigis","isoTauJets")

l1ExtraLayer2.etTotalSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
l1ExtraLayer2.etHadSource   = cms.InputTag("simCaloStage1LegacyFormatDigis")
l1ExtraLayer2.etMissSource  = cms.InputTag("simCaloStage1LegacyFormatDigis")
l1ExtraLayer2.htMissSource  = cms.InputTag("simCaloStage1LegacyFormatDigis")

l1ExtraLayer2.hfRingEtSumsSource    = cms.InputTag("simCaloStage1LegacyFormatDigis")
l1ExtraLayer2.hfRingBitCountsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")

## update l1ExtraLayer2 muon tag
l1ExtraLayer2.muonSource = cms.InputTag("simGmtDigis")


# the sequence
L1TCaloStage1_PPFromRaw = cms.Sequence(
    L1TRerunHCALTP_FromRAW
    +ecalDigis
    +simRctDigis
    +L1TCaloStage1
)
