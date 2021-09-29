import FWCore.ParameterSet.Config as cms

hltPFPuppiMET = cms.EDProducer("PFMETProducer",
    applyWeight = cms.bool(True),
    calculateSignificance = cms.bool(False),
    globalThreshold = cms.double(0.0),
    src = cms.InputTag("particleFlowTmp"),
    srcWeights = cms.InputTag("hltPFPuppiNoLep")
)
