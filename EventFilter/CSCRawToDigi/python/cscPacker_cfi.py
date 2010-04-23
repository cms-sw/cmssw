import FWCore.ParameterSet.Config as cms

cscpacker = cms.EDProducer("CSCDigiToRawModule",
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    preTriggerTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    requirePreTrigger = cms.bool(True),
    correlatedLCTDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")
)


