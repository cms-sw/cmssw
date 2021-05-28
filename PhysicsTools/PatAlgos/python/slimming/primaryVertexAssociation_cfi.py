import FWCore.ParameterSet.Config as cms
from CommonTools.RecoAlgos.sortedPFPrimaryVertices_cfi import sortedPFPrimaryVertices

primaryVertexAssociation = sortedPFPrimaryVertices.clone(
  qualityForPrimary = 2,
  produceSortedVertices = False,
  producePileUpCollection  = False,
  produceNoPileUpCollection = False
)

primaryVertexWithBSAssociation = primaryVertexAssociation.clone(
  vertices = "offlinePrimaryVerticesWithBS"
)

from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toModify(primaryVertexAssociation,particles = "cleanedParticleFlow")
primaryVertexAssociationCleaned = primaryVertexAssociation.clone(particles = "cleanedParticleFlow:removed")

run2_miniAOD_pp_on_AA_103X.toModify(primaryVertexWithBSAssociation,particles = "cleanedParticleFlow")
primaryVertexWithBSAssociationCleaned = primaryVertexWithBSAssociation.clone(particles = "cleanedParticleFlow:removed")
