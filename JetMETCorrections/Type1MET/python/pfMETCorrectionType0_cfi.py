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

from CommonTools.RecoUtils.pfcand_assomap_cfi import PFCandAssoMap as _PFCandAssoMap
pfCandidateToVertexAssociation = _PFCandAssoMap.clone(
    PFCandidateCollection = cms.InputTag('particleFlow'),
    UseBeamSpotCompatibility = cms.untracked.bool(True),
    ignoreMissingCollection = cms.bool(True)
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 0 MET corrections

pfMETcorrType0 = cms.EDProducer("Type0PFMETcorrInputProducer",
    srcPFCandidateToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociation'),
    srcHardScatterVertex = cms.InputTag('selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0'),
    correction = cms.PSet(
      # RunI correction
      #  formula = cms.string("-([0] + [1]*x)*(1.0 + TMath::Erf(-[2]*TMath::Power(x, [3])))"),
      #  par0 = cms.double(0.),
      #  par1 = cms.double(-0.703151),
      #  par2 = cms.double(0.0303531),
      #  par3 = cms.double(0.909209)  
        formula = cms.string("(x<35)?(-( [0]+x*[1]+pow(x, 2)*[2]+pow(x, 3)*[3] )):(-( [0]+35*[1]+pow(35, 2)*[2]+pow(35, 3)*[3] ))"),
        par0 = cms.double(-1.81414e-01),
        par1 = cms.double(-4.76934e-01),
        par2 = cms.double(8.63564e-03),
        par3 = cms.double(-4.94181e-05)      
    ),
    minDz = cms.double(0.2) # [cm], minimum distance required between pile-up vertices and "hard scatter" vertex
)   
#--------------------------------------------------------------------------------

type0PFMEtCorrectionPFCandToVertexAssociationTask = cms.Task(
    selectedVerticesForPFMEtCorrType0,
    selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0,
    particleFlowDisplacedVertex,
    pfCandidateToVertexAssociation
)

type0PFMEtCorrectionPFCandToVertexAssociation = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociationTask
)

type0PFMEtCorrectionPFCandToVertexAssociationForValidation = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociationTask
)

type0PFMEtCorrectionPFCandToVertexAssociationForValidationMiniAOD = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociationTask
)

type0PFMEtCorrectionTask = cms.Task(
    type0PFMEtCorrectionPFCandToVertexAssociationTask,
    pfMETcorrType0
)

type0PFMEtCorrection = cms.Sequence(
    type0PFMEtCorrectionTask
)
