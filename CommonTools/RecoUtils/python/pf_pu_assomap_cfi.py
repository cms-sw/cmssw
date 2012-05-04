import FWCore.ParameterSet.Config as cms

Tracks2Vertex = cms.EDProducer('PF_PU_AssoMap',	
	 
	  #Set the Input Collections
          VertexCollection = cms.InputTag('offlinePrimaryVertices'),
          TrackCollection = cms.InputTag('generalTracks'),
	   
	  #Configuration for the reassociation of gamma conversion particles
          GsfElectronCollection = cms.InputTag('gsfElectrons'),
	  ConversionsCollection = cms.InputTag('allConversions'),
	   
	  #Configuration for the reassociation of particles from V0 decays
	  V0KshortCollection = cms.InputTag('generalV0Candidates','Kshort'),
	  V0LambdaCollection = cms.InputTag('generalV0Candidates','Lambda'),
	   
	  #Configuration for the reassociation of particles from nuclear interactions
	  NIVertexCollection = cms.InputTag('particleFlowDisplacedVertex'),
	   	   
	  #Configuration to check if a secondary is compatible with the BeamSpot
	  UseBeamSpotCompatibility = cms.untracked.bool(False),
	  BeamSpot = cms.InputTag('offlineBeamSpot'),
		  
	  #Configuration for the final association
          VertexAssOneDim = cms.untracked.bool(True),
          VertexAssClosest = cms.untracked.bool(True),
          VertexAssUseAbsDistance = cms.untracked.bool(False),
)

