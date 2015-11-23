#
# simDigis for L1TGlobal package
#  

# 
# Define emulator sequence for producing L1TGlobal digis
# 

import FWCore.ParameterSet.Config as cms

## Run the Stage 2 uGT emulator
from L1Trigger.L1TGlobal.simGlobalStage2Digis_cff import *
simGlobalStage2Digis.caloInputTag = cms.InputTag('simCaloStage2Digis')
simGlobalStage2Digis.GmtInputTag = cms.InputTag('simGmtDigis')
simGlobalStage2Digis.PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv')
simGlobalStage2Digis.PrescaleSet = cms.uint32(1)
