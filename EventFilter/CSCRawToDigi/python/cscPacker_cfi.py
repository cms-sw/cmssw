import FWCore.ParameterSet.Config as cms

cscpacker = cms.EDProducer("CSCDigiToRawModule",
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    preTriggerTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    correlatedLCTDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"),
    # if min parameter = -999 always accept
    alctWindowMin = cms.int32(-3),
    alctWindowMax = cms.int32(3),
    clctWindowMin = cms.int32(-3),
    clctWindowMax = cms.int32(3),
    preTriggerWindowMin = cms.int32(-3),
    preTriggerWindowMax = cms.int32(1)
)


##
## Make changes for running in Run 2
##
# packer - simply get rid of it
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( cscpacker, useFormatVersion = cms.uint32(2013) )
run2_common.toModify( cscpacker, usePreTriggers = cms.bool(False) )
run2_common.toModify( cscpacker, packEverything = cms.bool(True) )
