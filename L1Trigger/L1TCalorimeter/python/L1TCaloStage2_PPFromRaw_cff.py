import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloStage2Params_cfi import *

# HCAL TP hack
from L1Trigger.L1TCalorimeter.L1TRerunHCALTP_FromRaw_cff import *

# stage 2
from L1Trigger.L1TCalorimeter.L1TCaloStage2_cff import *
caloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')

# the sequence
L1TCaloStage2_PPFromRaw = cms.Sequence(
    L1TRerunHCALTP_FromRAW
    +L1TCaloStage2
)
