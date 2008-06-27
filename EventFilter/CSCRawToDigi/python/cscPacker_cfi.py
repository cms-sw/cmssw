import FWCore.ParameterSet.Config as cms

cscpacker = cms.EDFilter("CSCDigiToRawModule",
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    correlatedLCTDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")
)


