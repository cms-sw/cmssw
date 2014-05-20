import FWCore.ParameterSet.Config as cms
#l1tStage1CaloParams_cfi.py
#from L1Trigger.L1TCalorimeter.l1tCaloStage1Params_cfi import *
from L1Trigger.L1TCalorimeter.l1tStage1CaloParams_cfi import *

from L1Trigger.L1TCalorimeter.l1tCaloStage1Digis_cfi import *
#from L1Trigger.L1TCalorimeter.l1tCaloStage2Layer1Digis_cfi import *

#from L1Trigger.L1TCalorimeter.l1tCaloStage2Digis_cfi import *

#content = cms.EDAnalyzer("EventContentAnalyzer")

L1TCaloStage1 = cms.Sequence(
    l1tCaloRCTToUpgradeConverter +
    l1tCaloStage1Digis +
    l1tPhysicalEtAdder +
    l1tCaloUpgradeToGCTConverter
)
