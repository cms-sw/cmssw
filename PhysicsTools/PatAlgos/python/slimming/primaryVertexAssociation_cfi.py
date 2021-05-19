import FWCore.ParameterSet.Config as cms
from CommonTools.RecoAlgos.sortedPFPrimaryVertices_cfi import sortedPFPrimaryVertices

primaryVertexAssociation = sortedPFPrimaryVertices.clone(
  qualityForPrimary = cms.int32(2),
  produceSortedVertices = cms.bool(False),
  producePileUpCollection  = cms.bool(False),
  produceNoPileUpCollection = cms.bool(False)
)

primaryVertexWithBSAssociation = primaryVertexAssociation.clone(
  vertices = cms.InputTag("offlinePrimaryVerticesWithBS")
)

from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toModify(primaryVertexAssociation,particles = "cleanedParticleFlow")
primaryVertexAssociationCleaned = primaryVertexAssociation.clone(particles = "cleanedParticleFlow:removed")

run2_miniAOD_pp_on_AA_103X.toModify(primaryVertexWithBSAssociation,particles = "cleanedParticleFlow")
primaryVertexWithBSAssociationCleaned = primaryVertexWithBSAssociation.clone(particles = "cleanedParticleFlow:removed")
