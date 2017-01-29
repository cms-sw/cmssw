import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.VertexCompositeCandidateCollectionSelector_cfi import VertexCompositeCandidateCollectionSelector

vertexCompositeCandidateCollectionSelector = VertexCompositeCandidateCollectionSelector.clone()
vertexCompositeCandidateCollectionSelector.v0            = cms.InputTag('generalV0Candidates:Kshort') # generalV0Candidates:Lambda
vertexCompositeCandidateCollectionSelector.beamSpot      = cms.InputTag('offlineBeamSpot')
vertexCompositeCandidateCollectionSelector.primaryVertex = cms.InputTag('offlinePrimaryVertices')
vertexCompositeCandidateCollectionSelector.pvNDOF        = cms.int32(4)
vertexCompositeCandidateCollectionSelector.lxyCUT        = cms.double( 16.) # cm (2016 pixel layer3:10.2 cm ; 2017 pixel layer4: 16.0 cm)
vertexCompositeCandidateCollectionSelector.lxyWRTbsCUT   = cms.double(  0.)  # cm
#vertexCompositeCandidateCollectionSelector.debug         = cms.untracked.bool(False)
