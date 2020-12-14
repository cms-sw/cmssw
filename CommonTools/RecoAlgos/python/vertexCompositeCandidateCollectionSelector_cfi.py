import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.VertexCompositeCandidateCollectionSelector_cfi import VertexCompositeCandidateCollectionSelector

vertexCompositeCandidateCollectionSelector = VertexCompositeCandidateCollectionSelector.clone(
    v0            = 'generalV0Candidates:Kshort', # generalV0Candidates:Lambda
    beamSpot      = 'offlineBeamSpot',
    primaryVertex = 'offlinePrimaryVertices',
    pvNDOF        = 4,
    lxyCUT        = 16., # cm (2016 pixel layer3:10.2 cm ; 2017 pixel layer4: 16.0 cm)
    lxyWRTbsCUT   = 0.,  # cm
    #debug         = cms.untracked.bool(False)
)
