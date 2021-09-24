import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.pfNoPileUpJME_cff import adapt, pfPileUpJME
from CommonTools.RecoAlgos.sortedPackedPrimaryVertices_cfi import sortedPackedPrimaryVertices

packedPrimaryVertexAssociationJME = sortedPackedPrimaryVertices.clone(
  produceSortedVertices = False,
  producePileUpCollection  = False,
  produceNoPileUpCollection = False
)
adapt(packedPrimaryVertexAssociationJME)

from CommonTools.ParticleFlow.pfNoPileUpPacked_cfi import pfNoPileUpPacked as _pfNoPileUpPacked
pfCHS = _pfNoPileUpPacked.clone(
    vertexAssociationQuality=pfPileUpJME.vertexAssociationQuality
)
