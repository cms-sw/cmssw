import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.pfNoPileUpJME_cff import primaryVertexAssociationJME

pfCHS = cms.EDFilter("CandPtrSelector",
              src = cms.InputTag("packedPFCandidates"),
              cut = cms.string("fromPV(0)>0"+\
	       " || (vertexRef().key<="+str(primaryVertexAssociationJME.assignment.NumOfPUVtxsForCharged.value())+" && "+\
		    "abs(dz(0))<"+str(primaryVertexAssociationJME.assignment.DzCutForChargedFromPUVtxs.value())+")"
                    )
)
