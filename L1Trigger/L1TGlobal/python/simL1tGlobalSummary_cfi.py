import FWCore.ParameterSet.Config as cms

###
### L1TGlobalSummary input parameters for when running uGT emulator
##

from L1Trigger.L1TGlobal.l1tGlobalSummary_cfi import *
l1tGlobalSummary.AlgInputTag = cms.InputTag("simGtStage2Digis")
l1tGlobalSummary.ExtInputTag = cms.InputTag("simGtExtFakeProd")
l1tGlobalSummary.MinBx          = cms.int32(0)
l1tGlobalSummary.MaxBx          = cms.int32(0)
