import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import pfPileUp as _pfPileUp
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import pfNoPileUp as _pfNoPileUp
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

def adapt(primaryVertexAssociationJME):
  # options for quality PrimaryDz = 6 (used in PUPPI)
  primaryVertexAssociationJME.assignment.maxDzSigForPrimaryAssignment = 1e10
  primaryVertexAssociationJME.assignment.maxDzForPrimaryAssignment = 0.3
  primaryVertexAssociationJME.assignment.maxDzErrorForPrimaryAssignment = 1e10
  primaryVertexAssociationJME.assignment.NumOfPUVtxsForCharged = 2
  primaryVertexAssociationJME.assignment.PtMaxCharged = 20.
  primaryVertexAssociationJME.assignment.EtaMinUseDz = 2.4
  primaryVertexAssociationJME.assignment.OnlyUseFirstDz = True
  from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
  phase2_common.toModify(
    primaryVertexAssociationJME.assignment,
    maxDzForPrimaryAssignment=0.1,
    EtaMinUseDz = 4.0
    )
primaryVertexAssociationJME = primaryVertexAssociation.clone(vertices = 'goodOfflinePrimaryVertices')
adapt(primaryVertexAssociationJME)

pfPileUpJME = _pfPileUp.clone(PFCandidates='particleFlowPtrs',
                              useVertexAssociation = True,
                              vertexAssociationQuality = 7,
                              vertexAssociation = ('primaryVertexAssociationJME','original'),
                              )
pfNoPileUpJME = _pfNoPileUp.clone(topCollection = 'pfPileUpJME',
                                  bottomCollection = 'particleFlowPtrs' )

pfNoPileUpJMETask = cms.Task(
    goodOfflinePrimaryVertices,
    primaryVertexAssociationJME,
    pfPileUpJME,
    pfNoPileUpJME
    )

pfNoPileUpJMESequence = cms.Sequence(pfNoPileUpJMETask)
