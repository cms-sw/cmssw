import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.pfNoPileUpJME_cff import adapt, pfPileUpJME
from CommonTools.RecoAlgos.sortedPackedPrimaryVertices_cfi import sortedPackedPrimaryVertices

packedPrimaryVertexAssociationJME = sortedPackedPrimaryVertices.clone(
  produceSortedVertices = False,
  producePileUpCollection  = False,
  produceNoPileUpCollection = False
)
adapt(packedPrimaryVertexAssociationJME)

pfCHS = cms.EDProducer("PFnoPileUpPacked",
  candidates = cms.InputTag("packedPFCandidates"),
  vertexAssociationQuality = pfPileUpJME.vertexAssociationQuality,
  vertexAssociation = cms.InputTag("packedPrimaryVertexAssociationJME","original")
  )
