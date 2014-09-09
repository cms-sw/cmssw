import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.l1tCaloStage2Params_cfi import *
from L1Trigger.L1TCalorimeter.l1tCaloStage2Layer1Digis_cfi import *
from L1Trigger.L1TCalorimeter.l1tCaloStage2Digis_cfi import *

#content = cms.EDAnalyzer("EventContentAnalyzer")

L1TCaloStage2 = cms.Sequence(
    l1tCaloStage2Layer1Digis +
    l1tCaloStage2Digis
)
