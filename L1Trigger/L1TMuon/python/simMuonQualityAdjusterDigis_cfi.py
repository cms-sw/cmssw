import FWCore.ParameterSet.Config as cms
import os

simMuonQualityAdjusterDigis = cms.EDProducer('L1TMuonQualityAdjuster',
    bmtfInput  = cms.InputTag("simBmtfDigis", "BMTF"),
    omtfInput  = cms.InputTag("simOmtfDigis", "OMTF"),
    emtfInput  = cms.InputTag("simEmtfDigis", "EMTF"),
    bmtfBxOffset  = cms.int32(0),
)





