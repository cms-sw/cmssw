import FWCore.ParameterSet.Config as cms

###
### L1TGlobalSummary input parameters for when running uGT emulator
##

from L1Trigger.L1TGlobal.L1TGlobalSummary_cfi import *
L1TGlobalSummary.AlgInputTag = cms.InputTag("simGtStage2Digis")
L1TGlobalSummary.ExtInputTag = cms.InputTag("simGtExtFakeProd")
L1TGlobalSummary.MinBx          = cms.int32(0)
L1TGlobalSummary.MaxBx          = cms.int32(0)
