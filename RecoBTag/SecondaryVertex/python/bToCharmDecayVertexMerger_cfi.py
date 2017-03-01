import FWCore.ParameterSet.Config as cms

bToCharmDecayVertexMerged = cms.EDProducer("BtoCharmDecayVertexMerger",
      primaryVertices  = cms.InputTag("offlinePrimaryVertices"),
      secondaryVertices = cms.InputTag("inclusiveSecondaryVerticesFiltered"),
      maxDRUnique = cms.double(0.4),
      maxvecSumIMifsmallDRUnique = cms.double(5.5),
      minCosPAtomerge = cms.double(0.99),
      maxPtreltomerge = cms.double(7777.0)
)
