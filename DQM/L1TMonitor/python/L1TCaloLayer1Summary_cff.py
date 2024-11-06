import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.CaloTriggerPrimitives_cff import *
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Summary_cfi import *
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import *
from DQM.L1TMonitor.L1TCaloLayer1Summary_cfi import *

dqmSimCaloStage2Layer1Summary = simCaloStage2Layer1Summary.clone(
    caloLayer1Regions = "caloLayer1Digis",
    backupRegionToken = 'dqmSimCaloStage2Layer1Digis',
)
dqmSimCaloStage2Layer1Digis = simCaloStage2Layer1Digis.clone(
    ecalToken = cms.InputTag("ecalDigis", "EcalTriggerPrimitives"),
    hcalToken = cms.InputTag("hcalDigis", "")
)

l1tCaloLayer1SummarySeq = cms.Sequence(dqmSimCaloStage2Layer1Digis * dqmSimCaloStage2Layer1Summary * l1tCaloLayer1Summary)
