import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# select collection of "good" collision vertices

selectedVerticesForPFMEtCorrType0 = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
    filter = cms.bool(False)                                          
)

selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0 = cms.EDFilter("PATSingleVertexSelector",
    mode = cms.string('firstVertex'),
    vertices = cms.InputTag('selectedVerticesForPFMEtCorrType0'),
    filter = cms.bool(False)                                                    
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# association of PFCandidates to vertices

from RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cfi import particleFlowDisplacedVertex
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

from CommonTools.RecoUtils.pf_pu_assomap_cfi import Tracks2Vertex
trackToVertexAssociation = Tracks2Vertex.clone()

from CommonTools.RecoUtils.pfcand_assomap_cfi import PFCandAssoMap
pfCandidateToVertexAssociation = PFCandAssoMap.clone(
    VertexTrackAssociationMap = cms.InputTag('trackToVertexAssociation')
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 0 MET corrections

pfMETcorrType0 = cms.EDProducer("Type0PFMETcorrInputProducer",
    srcPFCandidateToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociation'),
    srcHardScatterVertex = cms.InputTag('selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0'),
    correction = cms.PSet(
        formula = cms.string("-([0] + [1]*x)*(1.0 + TMath::Erf(-[2]*TMath::Power(x, [3])))"),
        par0 = cms.double(0.),
        par1 = cms.double(-0.710135),
        par2 = cms.double(0.0870503),
        par3 = cms.double(0.621243)          
    ),
    minDz = cms.double(0.2) # [cm], minimum distance required between pile-up vertices and "hard scatter" vertex
)   
#--------------------------------------------------------------------------------

type0PFMEtCorrectionPFCandToVertexAssociation = cms.Sequence(
    selectedVerticesForPFMEtCorrType0
   * selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0
   * particleFlowDisplacedVertex
   * trackToVertexAssociation
   * pfCandidateToVertexAssociation
)

type0PFMEtCorrection = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociation
   * pfMETcorrType0
)
