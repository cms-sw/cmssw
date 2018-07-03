import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TCalorimeter.caloStage1Params_cfi import *

# HCAL TP hack
from L1Trigger.L1TCalorimeter.L1TRerunHCALTP_FromRaw_cff import *

#now taken from GT:
#from L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff import *

from Configuration.StandardSequences.RawToDigi_Data_cff import ecalDigis

# RCT
# HCAL input would be from hcalDigis if hack not needed
from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )

# stage 1 itself
from L1Trigger.L1TCalorimeter.L1TCaloStage1_cff import *

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
