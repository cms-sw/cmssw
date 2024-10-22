import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.CaloTriggerPrimitives_cff import *
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Summary_cfi import *
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import *
from DQM.L1TMonitor.L1TCaloLayer1Summary_cfi import *

simCaloStage2Layer1Summary.caloLayer1Regions = cms.InputTag("caloLayer1Digis", "")
simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis", "EcalTriggerPrimitives")
simCaloStage2Layer1Digis.hcalToken = cms.InputTag("hcalDigis", "")

l1tCaloLayer1SummarySeq = cms.Sequence(simCaloStage2Layer1Digis * simCaloStage2Layer1Summary * l1tCaloLayer1Summary)
