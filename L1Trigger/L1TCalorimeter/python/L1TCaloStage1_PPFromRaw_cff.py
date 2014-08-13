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
rctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
rctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")

# GT
from L1Trigger.Configuration.SimL1Emulator_cff import simGtDigis
simGtDigis.GmtInputTag = 'gtDigis'
simGtDigis.GctInputTag = 'caloStage1LegacyFormatDigis'
simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )

# L1Extra
import L1Trigger.Configuration.L1Extra_cff
l1ExtraReEmul = L1Trigger.Configuration.L1Extra_cff.l1extraParticles.clone()
l1ExtraReEmul.isolatedEmSource    = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
l1ExtraReEmul.nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")

l1ExtraReEmul.forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
l1ExtraReEmul.centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
l1ExtraReEmul.tauJetSource     = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")

l1ExtraReEmul.etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis")
l1ExtraReEmul.etHadSource   = cms.InputTag("caloStage1LegacyFormatDigis")
l1ExtraReEmul.etMissSource  = cms.InputTag("caloStage1LegacyFormatDigis")
l1ExtraReEmul.htMissSource  = cms.InputTag("caloStage1LegacyFormatDigis")

l1ExtraReEmul.hfRingEtSumsSource    = cms.InputTag("caloStage1LegacyFormatDigis")
l1ExtraReEmul.hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis")

# the sequence
L1TCaloStage1_PPFromRaw = cms.Sequence(
    L1TRerunHCALTP_FromRAW
    +ecalDigis
    +simRctDigis
    +L1TCaloStage1
    +simGtDigis
    +l1ExtraReEmul
)
