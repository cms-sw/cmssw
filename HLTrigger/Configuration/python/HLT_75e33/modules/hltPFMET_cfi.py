import FWCore.ParameterSet.Config as cms

hltPFMET = cms.EDProducer("PFMETProducer",
    calculateSignificance = cms.bool(False),
    globalThreshold = cms.double(0.0),
    src = cms.InputTag("particleFlowTmp")
)
# foo bar baz
# 5e9kUGMO2XVCX
# TzTaOa0asuUFE
