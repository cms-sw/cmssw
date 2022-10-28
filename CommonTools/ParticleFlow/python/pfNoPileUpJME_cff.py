import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfPileUp_cfi  import pfPileUp as _pfPileUp
from CommonTools.ParticleFlow.TopProjectors.pfNoPileUp_cfi import pfNoPileUp as _pfNoPileUp
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation

# The following module implements vertex association for JME.
# It is not run by default to save computing time (but can be run to investigate alternative vertex identification criteria).
# Instead its parameters are used as input to faster implementations in pfPileUpJME and pfCHS and puppi.
primaryVertexAssociationJME = primaryVertexAssociation.clone(vertices = 'goodOfflinePrimaryVertices')
primaryVertexAssociationJME.assignment.maxDzSigForPrimaryAssignment = 1e10
primaryVertexAssociationJME.assignment.maxDzForPrimaryAssignment = 0.3
primaryVertexAssociationJME.assignment.maxDzErrorForPrimaryAssignment = 1e10
primaryVertexAssociationJME.assignment.NumOfPUVtxsForCharged = 2
primaryVertexAssociationJME.assignment.DzCutForChargedFromPUVtxs = 0.2
primaryVertexAssociationJME.assignment.PtMaxCharged = 20.
primaryVertexAssociationJME.assignment.EtaMinUseDz = 2.4
primaryVertexAssociationJME.assignment.OnlyUseFirstDz = True
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(
  primaryVertexAssociationJME.assignment,
  maxDzForPrimaryAssignment=0.1,
  EtaMinUseDz = 4.0
  )

pfPileUpJME = _pfPileUp.clone(PFCandidates='particleFlowPtrs',
                              Vertices = 'goodOfflinePrimaryVertices',
                              checkClosestZVertex = False,
                              NumOfPUVtxsForCharged = primaryVertexAssociationJME.assignment.NumOfPUVtxsForCharged,
                              DzCutForChargedFromPUVtxs = primaryVertexAssociationJME.assignment.DzCutForChargedFromPUVtxs,
                              )
pfNoPileUpJME = _pfNoPileUp.clone(topCollection = 'pfPileUpJME',
                                  bottomCollection = 'particleFlowPtrs' )

pfNoPileUpJMETask = cms.Task(
    goodOfflinePrimaryVertices,
    pfPileUpJME,
    pfNoPileUpJME
    )

pfNoPileUpJMESequence = cms.Sequence(pfNoPileUpJMETask)
