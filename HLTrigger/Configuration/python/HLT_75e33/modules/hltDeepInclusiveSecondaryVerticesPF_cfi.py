import FWCore.ParameterSet.Config as cms

hltDeepInclusiveSecondaryVerticesPF = cms.EDProducer("CandidateVertexMerger",
    maxFraction = cms.double(0.7),
    minSignificance = cms.double(2),
    secondaryVertices = cms.InputTag("hltDeepInclusiveVertexFinderPF")
)
