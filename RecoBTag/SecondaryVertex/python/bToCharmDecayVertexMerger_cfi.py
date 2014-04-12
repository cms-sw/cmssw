import FWCore.ParameterSet.Config as cms

bToCharmDecayVertexMerged = cms.EDProducer("BtoCharmDecayVertexMerger",
      primaryVertices  = cms.InputTag("offlinePrimaryVertices"),
      secondaryVertices = cms.InputTag("inclusiveSecondaryVerticesFiltered"),
      minDRUnique = cms.untracked.double(0.4),
      minvecSumIMifsmallDRUnique = cms.untracked.double(5.5),
      minCosPAtomerge = cms.untracked.double(0.99),
      maxPtreltomerge = cms.untracked.double(7777.0)
)
